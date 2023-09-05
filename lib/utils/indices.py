# Taken from: https://meshcapade.wiki/SMPL
# The whole head
HEAD = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
    98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
    117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
    136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 154, 155, 156, 157, 158,
    159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 176, 177, 178, 179,
    180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198,
    199, 200, 201, 202, 203, 204, 205, 220, 221, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234,
    235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253,
    254, 255, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274,
    275, 276, 277, 278, 279, 280, 281, 282, 283, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295,
    303, 304, 306, 307, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324,
    325, 326, 327, 328, 329, 330, 331, 332, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345,
    346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364,
    365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383,
    384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402,
    403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421,
    422, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 442, 443, 444, 445, 446,
    447, 448, 449, 450, 454, 455, 456, 457, 458, 459, 461, 462, 463, 464, 465, 466, 467, 468, 469,
    470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488,
    489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507,
    508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526,
    527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545,
    546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564,
    565, 566, 567, 568, 569, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 1764, 1765, 1766,
    1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1905, 1906, 1907, 1908, 2779, 2780, 2781,
    2782, 2783, 2784, 2785, 2786, 2787, 2788, 2789, 2790, 2791, 2792, 2793, 2794, 2795, 2796, 2797,
    2798, 2799, 2800, 2801, 2802, 2803, 2804, 2805, 2806, 2807, 2808, 2809, 2810, 2811, 2814, 2815,
    2816, 2817, 2818, 3045, 3046, 3047, 3048, 3051, 3052, 3053, 3054, 3055, 3056, 3058, 3069, 3070,
    3071, 3072, 3161, 3162, 3163, 3165, 3166, 3167, 3485, 3486, 3487, 3488, 3489, 3490, 3491, 3492,
    3493, 3494, 3499, 3512, 3513, 3514, 3515, 3516, 3517, 3518, 3519, 3520, 3521, 3522, 3523, 3524,
    3525, 3526, 3527, 3528, 3529, 3530, 3531, 3532, 3533, 3534, 3535, 3536, 3537, 3538, 3539, 3540,
    3541, 3542, 3543, 3544, 3545, 3546, 3547, 3548, 3549, 3550, 3551, 3552, 3553, 3554, 3555, 3556,
    3557, 3558, 3559, 3560, 3561, 3562, 3563, 3564, 3565, 3566, 3567, 3568, 3569, 3570, 3571, 3572,
    3573, 3574, 3575, 3576, 3577, 3578, 3579, 3580, 3581, 3582, 3583, 3584, 3585, 3586, 3587, 3588,
    3589, 3590, 3591, 3592, 3593, 3594, 3595, 3596, 3597, 3598, 3599, 3600, 3601, 3602, 3603, 3604,
    3605, 3606, 3607, 3608, 3609, 3610, 3611, 3612, 3613, 3614, 3615, 3616, 3617, 3618, 3619, 3620,
    3621, 3622, 3623, 3624, 3625, 3626, 3627, 3628, 3629, 3630, 3631, 3632, 3633, 3634, 3635, 3636,
    3637, 3638, 3639, 3640, 3641, 3642, 3643, 3644, 3645, 3646, 3647, 3648, 3649, 3650, 3651, 3652,
    3653, 3654, 3655, 3656, 3657, 3658, 3659, 3660, 3661, 3666, 3667, 3668, 3669, 3670, 3671, 3672,
    3673, 3674, 3675, 3676, 3677, 3678, 3679, 3680, 3681, 3682, 3683, 3684, 3685, 3688, 3689, 3690,
    3691, 3692, 3693, 3694, 3695, 3696, 3697, 3698, 3699, 3700, 3701, 3702, 3703, 3704, 3705, 3706,
    3707, 3708, 3709, 3710, 3711, 3712, 3713, 3714, 3715, 3716, 3717, 3732, 3733, 3737, 3738, 3739,
    3740, 3741, 3742, 3743, 3744, 3745, 3746, 3747, 3748, 3749, 3750, 3751, 3752, 3753, 3754, 3755,
    3756, 3757, 3758, 3759, 3760, 3761, 3762, 3763, 3764, 3765, 3766, 3767, 3770, 3771, 3772, 3773,
    3774, 3775, 3776, 3777, 3778, 3779, 3780, 3781, 3782, 3783, 3784, 3785, 3786, 3787, 3788, 3789,
    3790, 3791, 3792, 3793, 3794, 3795, 3798, 3799, 3800, 3801, 3802, 3803, 3804, 3805, 3806, 3807,
    3815, 3816, 3819, 3820, 3821, 3822, 3823, 3824, 3825, 3826, 3827, 3828, 3829, 3830, 3831, 3832,
    3833, 3834, 3835, 3836, 3837, 3838, 3841, 3842, 3843, 3844, 3845, 3846, 3847, 3848, 3849, 3850,
    3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859, 3860, 3861, 3862, 3863, 3864, 3865, 3866,
    3867, 3868, 3869, 3870, 3871, 3872, 3873, 3874, 3875, 3876, 3877, 3878, 3879, 3880, 3881, 3882,
    3883, 3884, 3885, 3886, 3887, 3888, 3889, 3890, 3891, 3892, 3893, 3894, 3895, 3896, 3897, 3898,
    3899, 3900, 3901, 3902, 3903, 3904, 3905, 3906, 3907, 3908, 3909, 3910, 3911, 3912, 3913, 3914,
    3915, 3916, 3917, 3922, 3923, 3924, 3925, 3926, 3927, 3928, 3929, 3930, 3931, 3932, 3933, 3936,
    3937, 3938, 3939, 3940, 3941, 3945, 3946, 3947, 3948, 3949, 3950, 3951, 3952, 3953, 3954, 3955,
    3956, 3957, 3958, 3959, 3960, 3961, 3962, 3963, 3964, 3965, 3966, 3967, 3968, 3969, 3970, 3971,
    3972, 3973, 3974, 3975, 3976, 3977, 3978, 3979, 3980, 3981, 3982, 3983, 3984, 3985, 3986, 3987,
    3988, 3989, 3990, 3991, 3992, 3993, 3994, 3995, 3996, 3997, 3998, 3999, 4000, 4001, 4002, 4003,
    4004, 4005, 4006, 4007, 4008, 4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017, 4018, 4019,
    4020, 4021, 4022, 4023, 4024, 4025, 4026, 4027, 4028, 4029, 4030, 4031, 4032, 4033, 4034, 4035,
    4036, 4037, 4038, 4039, 4040, 4041, 4042, 4043, 4044, 4045, 4046, 4047, 4048, 4049, 4050, 4051,
    4052, 4053, 4054, 4055, 4056, 4057, 4062, 4063, 4064, 4065, 4066, 4067, 4068, 4069, 4070, 4071,
    5231, 5232, 5233, 5235, 5236, 5237, 5238, 5239, 5240, 5241, 5242, 5243, 5366, 5367, 5368, 5369,
    6240, 6241, 6242, 6243, 6244, 6245, 6246, 6247, 6248, 6249, 6250, 6251, 6252, 6253, 6254, 6255,
    6256, 6257, 6258, 6259, 6260, 6261, 6262, 6263, 6264, 6265, 6266, 6267, 6268, 6269, 6270, 6271,
    6272, 6275, 6276, 6277, 6278, 6279, 6492, 6493, 6494, 6495, 6880, 6881, 6882, 6883, 6884, 6885,
    6886, 6887, 6888, 6889
]

FACE = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
    98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
    117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
    136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 154, 155, 156, 157, 188, 189, 190, 191,
    192, 193, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243,
    244, 245, 246, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 275, 276, 277, 278, 279,
    280, 281, 282, 283, 286, 287, 288, 289, 290, 291, 292, 293, 294, 310, 311, 312, 313, 314, 315,
    316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 335, 336,
    337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355,
    356, 357, 358, 359, 6495, 377, 392, 405, 407, 408, 409, 410, 415, 416, 417, 418, 419, 420, 421,
    427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 442, 443, 454, 455, 456, 458,
    459, 462, 6240, 6241, 6242, 6243, 6244, 6245, 6246, 6247, 6248, 6249, 6250, 6251, 2779, 2780,
    2781, 2782, 6252, 2783, 2784, 2785, 2786, 6253, 2787, 2788, 2789, 2790, 6254, 2791, 2792, 2793,
    2794, 6255, 2795, 2796, 2797, 2798, 6256, 2799, 2800, 2801, 2802, 6257, 2803, 2804, 2805, 2806,
    6258, 2807, 2808, 2809, 2810, 6259, 2811, 2814, 2815, 2816, 6260, 2817, 2818, 6261, 6262, 6263,
    6264, 6265, 6266, 6267, 6268, 6269, 6270, 6271, 6272, 6275, 6276, 6277, 6278, 6279, 3045, 3046,
    3047, 3048, 3051, 3052, 3053, 3054, 3055, 3056, 3069, 3070, 3071, 3072, 3161, 3162, 3163, 3165,
    3166, 3167, 5366, 5367, 5368, 5369, 3499, 3512, 3513, 3514, 3515, 3516, 3517, 3518, 3519, 3520,
    3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528, 3529, 3530, 3531, 3532, 3533, 3534, 3535, 3536,
    3537, 3538, 3539, 3540, 3541, 3542, 3543, 3544, 3545, 3546, 3547, 3548, 3549, 3550, 3551, 3552,
    3553, 3554, 3555, 3556, 3557, 3558, 3559, 3560, 3561, 3562, 3563, 3564, 3565, 3566, 3567, 3568,
    3569, 3570, 3571, 3572, 3573, 3574, 3575, 3576, 3577, 3578, 3579, 3580, 3581, 3582, 3583, 3584,
    3585, 3586, 3587, 3588, 3589, 3590, 3591, 3592, 3593, 3594, 3595, 3596, 3597, 3598, 3599, 3600,
    3601, 3602, 3603, 3604, 3605, 3606, 3607, 3608, 3609, 3610, 3611, 3612, 3613, 3614, 3615, 3616,
    3617, 3618, 3619, 3620, 3621, 3622, 3623, 3624, 3625, 3626, 3627, 3628, 3629, 3630, 3631, 3632,
    3633, 3634, 3635, 3636, 3637, 3638, 3639, 3640, 3641, 3642, 3643, 3644, 3645, 3646, 3647, 3648,
    3649, 3650, 3651, 3652, 3653, 3654, 3655, 3656, 3657, 3658, 3666, 3667, 3668, 3669, 3700, 3701,
    3702, 3703, 3704, 3705, 3739, 3740, 3741, 3742, 3743, 3744, 3745, 3746, 3747, 3748, 3749, 3750,
    3751, 3752, 3753, 3754, 3755, 3756, 3757, 3758, 3770, 3771, 3772, 3773, 3774, 3775, 3776, 3777,
    3778, 3779, 3780, 3787, 3788, 3789, 3790, 3791, 3792, 3793, 3794, 3795, 3798, 3799, 3800, 3801,
    3802, 3803, 3804, 3805, 3806, 3820, 3821, 3822, 3823, 3824, 3825, 3826, 3827, 3828, 3829, 3830,
    3831, 3832, 3833, 3834, 3835, 3836, 3837, 3838, 3841, 3842, 3843, 3844, 3845, 3846, 3847, 3848,
    3849, 3850, 3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859, 3860, 3861, 3862, 3863, 3881,
    3894, 3908, 3909, 3910, 3911, 3912, 3913, 3914, 3915, 3916, 3922, 3923, 3924, 3925, 3926, 3927,
    3928, 3929, 3930, 3931, 3932, 3933, 3936, 3937, 3945, 3946, 3947, 3948, 3949, 3951, 1905, 1906,
    1907, 1908, 6492, 6493, 6494
]

FACE_CHEEKS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
    98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
    117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
    136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 154, 155, 156, 157, 160, 161, 188,
    189, 190, 191, 192, 193, 194, 195, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238,
    239, 240, 241, 242, 243, 244, 245, 246, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268,
    275, 276, 277, 278, 279, 280, 281, 282, 283, 286, 287, 288, 289, 290, 291, 292, 293, 294, 310,
    311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329,
    330, 331, 332, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350,
    351, 352, 353, 354, 355, 356, 357, 358, 359, 375, 376, 377, 378, 379, 380, 381, 382, 392, 393,
    402, 405, 406, 407, 408, 409, 410, 415, 416, 417, 418, 419, 420, 421, 427, 428, 429, 430, 431,
    432, 433, 434, 435, 436, 437, 438, 439, 442, 443, 447, 450, 454, 455, 456, 458, 459, 462, 472,
    484, 517, 1905, 1906, 1907, 1908, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2786, 2787, 2788,
    2789, 2790, 2791, 2792, 2793, 2794, 2795, 2796, 2797, 2798, 2799, 2800, 2801, 2802, 2803, 2804,
    2805, 2806, 2807, 2808, 2809, 2810, 2811, 2814, 2815, 2816, 2817, 2818, 3045, 3046, 3047, 3048,
    3051, 3052, 3053, 3054, 3055, 3056, 3069, 3070, 3071, 3072, 3161, 3162, 3163, 3165, 3166, 3167,
    3499, 3512, 3513, 3514, 3515, 3516, 3517, 3518, 3519, 3520, 3521, 3522, 3523, 3524, 3525, 3526,
    3527, 3528, 3529, 3530, 3531, 3532, 3533, 3534, 3535, 3536, 3537, 3538, 3539, 3540, 3541, 3542,
    3543, 3544, 3545, 3546, 3547, 3548, 3549, 3550, 3551, 3552, 3553, 3554, 3555, 3556, 3557, 3558,
    3559, 3560, 3561, 3562, 3563, 3564, 3565, 3566, 3567, 3568, 3569, 3570, 3571, 3572, 3573, 3574,
    3575, 3576, 3577, 3578, 3579, 3580, 3581, 3582, 3583, 3584, 3585, 3586, 3587, 3588, 3589, 3590,
    3591, 3592, 3593, 3594, 3595, 3596, 3597, 3598, 3599, 3600, 3601, 3602, 3603, 3604, 3605, 3606,
    3607, 3608, 3609, 3610, 3611, 3612, 3613, 3614, 3615, 3616, 3617, 3618, 3619, 3620, 3621, 3622,
    3623, 3624, 3625, 3626, 3627, 3628, 3629, 3630, 3631, 3632, 3633, 3634, 3635, 3636, 3637, 3638,
    3639, 3640, 3641, 3642, 3643, 3644, 3645, 3646, 3647, 3648, 3649, 3650, 3651, 3652, 3653, 3654,
    3655, 3656, 3657, 3658, 3659, 3666, 3667, 3668, 3669, 3671, 3672, 3700, 3701, 3702, 3703, 3704,
    3705, 3706, 3707, 3739, 3740, 3741, 3742, 3743, 3744, 3745, 3746, 3747, 3748, 3749, 3750, 3751,
    3752, 3753, 3754, 3755, 3756, 3757, 3758, 3770, 3771, 3772, 3773, 3774, 3775, 3776, 3777, 3778,
    3779, 3780, 3787, 3788, 3789, 3790, 3791, 3792, 3793, 3794, 3795, 3798, 3799, 3800, 3801, 3802,
    3803, 3804, 3805, 3806, 3820, 3821, 3822, 3823, 3824, 3825, 3826, 3827, 3828, 3829, 3830, 3831,
    3832, 3833, 3834, 3835, 3836, 3837, 3838, 3841, 3842, 3843, 3844, 3845, 3846, 3847, 3848, 3849,
    3850, 3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859, 3860, 3861, 3862, 3863, 3878, 3879,
    3881, 3882, 3883, 3884, 3885, 3886, 3894, 3895, 3906, 3907, 3908, 3909, 3910, 3911, 3912, 3913,
    3914, 3915, 3916, 3922, 3923, 3924, 3925, 3926, 3927, 3928, 3929, 3930, 3931, 3932, 3933, 3936,
    3937, 3938, 3939, 3945, 3946, 3947, 3948, 3949, 3951, 3960, 3972, 4005, 5366, 5367, 5368, 5369,
    6240, 6241, 6242, 6243, 6244, 6245, 6246, 6247, 6248, 6249, 6250, 6251, 6252, 6253, 6254, 6255,
    6256, 6257, 6258, 6259, 6260, 6261, 6262, 6263, 6264, 6265, 6266, 6267, 6268, 6269, 6270, 6271,
    6272, 6275, 6276, 6277, 6278, 6279, 6492, 6493, 6494, 6495
]

# 227: Nose, 521: right cheek, 166: left cheek, 348: between eye brows, 229: forehead, 355: right forehead, 2: left forehead
# 419: right corner of mouth, 71: left corner of mouth, 362: right edge, 10: left edge,
# 335: chin
FACE_UNIFORM_POINTS = [227, 521, 166, 348, 229, 355, 2, 419, 71, 362, 10, 335]

# 232: Nose,
FACE_CHEEKS_UNIFORM_POINTS = [1, 13, 51, 159, 172, 180, 186, 232, 234, 336, 356, 368, 375, 380, 387, 422, 530, 546, 552, 697] # [1, 172, 186, 232, 234, 336, 368, 375, 380, 546, 697]

EYE_MASK = [
    1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23, 104, 105, 106, 107, 108, 109, 110, 111, 112,
    113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
    136, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165,
    170, 171, 172, 176, 179, 180, 181, 182, 224, 225, 230, 232, 233, 234, 236, 237, 238, 239, 240,
    241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 257, 258, 259, 260,
    267, 276, 277, 283, 284, 285, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305,
    306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324,
    325, 326, 327, 328, 329, 330, 348, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 372, 373,
    374, 375, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472,
    473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 488, 500, 501, 502, 503, 504, 505, 506,
    507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 522, 523, 524, 528, 531, 532, 534, 535,
    576, 578, 579, 580, 581, 582, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596,
    597, 598, 599, 600, 601, 603, 610, 619, 620, 625, 626, 627, 635, 636, 637, 638, 639, 640, 641,
    642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660,
    661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672
]

EYE_MASK_FACE_CHEEKS = [
    1, 2, 4, 5, 6, 8, 10, 11, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
    118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 150, 154, 155, 156, 157,
    158, 163, 164, 165, 166, 167, 168, 169, 170, 175, 176, 177, 181, 186, 187, 235, 237, 238, 241,
    242, 243, 244, 245, 246, 247, 249, 250, 251, 254, 255, 256, 257, 258, 273, 275, 282, 291, 292,
    300, 301, 302, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328,
    329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347,
    348, 349, 350, 368, 374, 375, 376, 377, 379, 380, 381, 382, 476, 477, 478, 479, 480, 481, 482,
    483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501,
    502, 503, 523, 526, 527, 528, 529, 531, 535, 536, 537, 538, 539, 540, 541, 542, 547, 548, 549,
    553, 559, 560, 603, 604, 606, 607, 609, 610, 611, 612, 613, 615, 616, 617, 619, 620, 622, 623,
    624, 645, 654, 655, 662, 663, 664, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686,
    687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705,
    706, 707, 708, 709, 710, 711, 712
]