import os
import cv2
import numpy as np
import pyamg
import glob
from PIL import Image

from skimage.segmentation import find_boundaries
import scipy.sparse.linalg
import scipy.signal
import scipy.linalg
import scipy.sparse

class PoissonSeamlessCloner:
    #@profile
    def __init__(self, source, mask, target, solver='spsolve', scale=1.0):
        self.mask = mask / 255
        self.src_rgb = source[:, :, :3] / 255
        self.target_rgb = target[:, :, :3] / 255
        
        self.solver = solver
        if solver != "multigrid":
            self.solver_func = getattr(scipy.sparse.linalg, solver)
        else:
            self.solver_func = None

        self.img_h, self.img_w = self.mask.shape

        _, self.mask = cv2.threshold(self.mask, 0.5, 1, cv2.THRESH_BINARY) # fix here
        self.inner_mask, self.boundary_mask = process_mask(self.mask)
        
        self.pixel_ids = get_pixel_ids(self.mask) 
        self.inner_ids = get_masked_values(self.pixel_ids, self.inner_mask).flatten()
        self.boundary_ids = get_masked_values(self.pixel_ids, self.boundary_mask).flatten()
        self.mask_ids = get_masked_values(self.pixel_ids, self.mask).flatten() # boundary + inner
        
        self.inner_pos = np.searchsorted(self.mask_ids, self.inner_ids) 
        self.boundary_pos = np.searchsorted(self.mask_ids, self.boundary_ids)
        self.mask_pos = np.searchsorted(self.pixel_ids.flatten(), self.mask_ids)

    #@profile
    def construct_C_matrix(self):
        n1_pos = np.searchsorted(self.mask_ids, self.inner_ids - 1)
        n2_pos = np.searchsorted(self.mask_ids, self.inner_ids + 1)
        n3_pos = np.searchsorted(self.mask_ids, self.inner_ids - self.img_w)
        n4_pos = np.searchsorted(self.mask_ids, self.inner_ids + self.img_w)

        l = len(self.mask_ids)
        row_ids = np.concatenate([
            self.inner_pos, self.inner_pos, self.inner_pos, self.inner_pos, self.inner_pos, self.boundary_pos,
            l + self.inner_pos, l + self.inner_pos, l + self.inner_pos, l + self.inner_pos, l + self.inner_pos, l + self.boundary_pos,
            2 * l + self.inner_pos, 2 * l + self.inner_pos, 2 * l + self.inner_pos, 2 * l + self.inner_pos, 2 * l + self.inner_pos, 2 * l + self.boundary_pos
        ])
        col_ids = np.concatenate([
            n1_pos, n2_pos, n3_pos, n4_pos, self.inner_pos, self.boundary_pos,
            l + n1_pos, l + n2_pos, l + n3_pos, l + n4_pos, l + self.inner_pos, l + self.boundary_pos,
            2 * l + n1_pos, 2 * l + n2_pos, 2 * l + n3_pos, 2 * l + n4_pos, 2 * l + self.inner_pos, 2 * l + self.boundary_pos
        ])
        data = ([1] * len(self.inner_pos) * 4 + [-4] * len(self.inner_pos) + [1] * len(self.boundary_pos)) * 3
        C = scipy.sparse.csr_matrix((data, (row_ids, col_ids)), shape=(3 * len(self.mask_ids), 3 * len(self.mask_ids)))

        return C

    def construct_A_matrix(self):

        n1_pos = np.searchsorted(self.mask_ids, self.inner_ids - 1)
        n2_pos = np.searchsorted(self.mask_ids, self.inner_ids + 1)
        n3_pos = np.searchsorted(self.mask_ids, self.inner_ids - self.img_w)
        n4_pos = np.searchsorted(self.mask_ids, self.inner_ids + self.img_w)

        A = scipy.sparse.lil_matrix((len(self.mask_ids), len(self.mask_ids)))
        A[self.inner_pos, n1_pos] = 1
        A[self.inner_pos, n2_pos] = 1
        A[self.inner_pos, n3_pos] = 1
        A[self.inner_pos, n4_pos] = 1
        A[self.inner_pos, self.inner_pos] = -4 
        A[self.boundary_pos, self.boundary_pos] = 1
        A = A.tocsr()
        
        return A
    
    def construct_b(self, inner_gradient_values, boundary_pixel_values):
        b = np.zeros(len(self.mask_ids))
        b[self.inner_pos] = inner_gradient_values
        b[self.boundary_pos] = boundary_pixel_values
        return b
    
    #@profile
    def compute_mixed_gradients(self, src, target, mode="max", alpha=1.0):
        if mode == "max":
            Ix_src, Iy_src = compute_gradient(src)
            Ix_target, Iy_target = compute_gradient(target)
            I_src_amp = (Ix_src**2 + Iy_src**2)**0.5
            I_target_amp = (Ix_target**2 + Iy_target**2)**0.5
            Ix = np.where(I_src_amp > I_target_amp, Ix_src, Ix_target)
            Iy = np.where(I_src_amp > I_target_amp, Iy_src, Iy_target)
            Ixx, _ = compute_gradient(Ix, forward=False)
            _, Iyy = compute_gradient(Iy, forward=False)
            return Ixx + Iyy
        elif mode == "alpha":
            src_laplacian = compute_laplacian(src)
            target_laplacian = compute_laplacian(target)
            return alpha * src_laplacian + (1 - alpha) * target_laplacian
        else:
            raise ValueError(f"Gradient mixing mode '{mode}' not supported!")
    
    #@profile
    def poisson_blend_channel(self, src, target, gradient_mixing_mode, gradient_mixing_alpha):
        mixed_gradients = self.compute_mixed_gradients(src, target, gradient_mixing_mode, gradient_mixing_alpha)

        boundary_pixel_values = get_masked_values(target, self.boundary_mask).flatten()
        inner_gradient_values = get_masked_values(mixed_gradients, self.inner_mask).flatten()

        # Construct b
        b = self.construct_b(inner_gradient_values, boundary_pixel_values)

        # Solve Ax = b
        if self.solver != "multigrid":
            x = self.solver_func(self.A, b)
            if isinstance(x, tuple): # solvers other than spsolve
                x = x[0]
        else:
            # Use multigrid solver
            ml = pyamg.ruge_stuben_solver(self.A)
            x = ml.solve(b, tol=1e-10)
        new_src = np.zeros(src.size)
        new_src[self.mask_pos] = x
        new_src = new_src.reshape(src.shape)
        poisson_blended_img = get_alpha_blended_img(new_src, target, self.mask)

        poisson_blended_img = np.clip(poisson_blended_img, 0, 1)
        
        return poisson_blended_img
    
    #@profile
    def poisson_blend_rgb_v2(self, gradient_mixing_mode, gradient_mixing_alpha):
        self.C = self.construct_C_matrix()
        b_full = []
        for i in range(self.src_rgb.shape[-1]):
            src = self.src_rgb[..., i]
            target = self.target_rgb[..., i]
            mixed_gradients = self.compute_mixed_gradients(src, target, gradient_mixing_mode, gradient_mixing_alpha)
            boundary_pixel_values = get_masked_values(target, self.boundary_mask).flatten()
            inner_gradient_values = get_masked_values(mixed_gradients, self.inner_mask).flatten()
            b = self.construct_b(inner_gradient_values, boundary_pixel_values)
            b_full.append(b)
        b_full = np.concatenate(b_full)
        x = self.solver(self.C, b_full)#[0]
        x = x.reshape(3, -1).T
        new_src = np.zeros((self.img_w * self.img_h, 3))
        new_src[self.mask_pos, :] = x
        new_src = new_src.reshape(self.src_rgb.shape)
        poisson_blended_img = get_alpha_blended_img(new_src, self.target_rgb, np.expand_dims(self.mask, -1))

        poisson_blended_img = np.clip(poisson_blended_img, 0, 1)
        
        return poisson_blended_img

    #@profile
    def poisson_blend_rgb(self, gradient_mixing_mode, gradient_mixing_alpha):
        self.A = self.construct_A_matrix()
        poisson_blended_img_rgb = []
        for i in range(self.src_rgb.shape[-1]):
            poisson_blended_img_rgb.append(
                self.poisson_blend_channel(
                    self.src_rgb[..., i], self.target_rgb[..., i],
                    gradient_mixing_mode, gradient_mixing_alpha
                )
            )
        return np.dstack(poisson_blended_img_rgb)
    
    def poisson_blend_gray(self, gradient_mixing_mode, gradient_mixing_alpha):
        self.A = self.construct_A_matrix()
        src_gray = rgb2gray(self.src_rgb)
        target_gray = rgb2gray(self.target_rgb)
        return self.poisson_blend_channel(src_gray, target_gray, gradient_mixing_mode, gradient_mixing_alpha)



def blend(source, mask, target, alpha_value):
    cloner = PoissonSeamlessCloner(source, mask, target)
    img = cloner.poisson_blend_rgb('alpha', alpha_value)
    img = (img * 255).astype(np.uint8)
    return img


def read_image(folder, name, scale=1, gray=False):
    for filename in glob.glob(folder + "/*"):
        if os.path.splitext(os.path.basename(filename))[0] == name:
            break
    img = Image.open(os.path.join(filename))
    if scale != 1:
        w, h = img.size
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    if gray:
        img = img.convert("L")
    img = np.array(img)
    if len(img.shape) == 3:
        img = img[..., :3]
    return img.astype(np.float64) / 255 # only first 3

def rgb2gray(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def process_mask(mask):
    boundary = find_boundaries(mask, mode="inner").astype(int)
    inner = mask - boundary
    return inner, boundary

def compute_laplacian(img):
    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
    laplacian = scipy.signal.fftconvolve(img, kernel, mode="same")
    return laplacian

def compute_gradient(img, forward=True):
    if forward:
        kx = np.array([
            [0, 0, 0],
            [0, -1, 1],
            [0, 0, 0]
        ])
        ky = np.array([
            [0, 0, 0],
            [0, -1, 0],
            [0, 1, 0]
        ])
    else:
        kx = np.array([
            [0, 0, 0],
            [-1, 1, 0],
            [0, 0, 0]
        ])
        ky = np.array([
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
    Gx = scipy.signal.fftconvolve(img, kx, mode="same")
    Gy = scipy.signal.fftconvolve(img, ky, mode="same")
    return Gx, Gy

def get_pixel_ids(img):
    pixel_ids = np.arange(img.shape[0] * img.shape[1]).reshape(img.shape[0], img.shape[1])
    return pixel_ids

def get_masked_values(values, mask):
    assert values.shape == mask.shape
    nonzero_idx = np.nonzero(mask) # get mask 1
    return values[nonzero_idx]

def get_alpha_blended_img(src, target, alpha_mask):
    return src * alpha_mask + target * (1 - alpha_mask)

def dilate_img(img, k):
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(img, kernel, iterations = 1)

def estimate_sparse_rank(A):
    def mv(v):
        return A @ v
    L = scipy.sparse.linalg.LinearOperator(A.shape, matvec=mv, rmatvec=mv)
    rank = scipy.linalg.interpolative.estimate_rank(L, 0.1)
    return rank

