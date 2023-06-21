import numpy as np
import h5py

path1 = "/workspace/koina/test/Prosit/20230621185559_prediction.hdf5"
data1 = h5py.File(path1, "r+")
intensities_pred = np.array(data1["intensities_pred"][:])

array = intensities_pred[0:5]

np.save(
    "/workspace/koina/test/Prosit/arr_Prosit_2023_intensity_XL_CMS2_int_raw.npy", array
)

print(array)

"""

# seq = np.load(
# "/workspace/koina/test/Prosit/arr_Prosit_2023_intensity_XL_CMS2_seq_1.npy"
# )
charge = np.load(
    "/workspace/koina/test/Prosit/arr_Prosit_2023_intensity_XL_CMS2_charge.npy"
)
ces = np.load("/workspace/koina/test/Prosit/arr_Prosit_2023_intensity_XL_CMS2_ces.npy")
int_raw = np.load(
    "/workspace/koina/test/Prosit/arr_Prosit_2020_intensity_hcd_int_raw.npy"
)
# int = np.load("arr_Prosit_2020_intensity_hcd_int.npy")

# print(seq)
# (charge)
# print(ces)
print(int_raw)
# print(int)


# np.save(
# "/workspace/koina/test/Prosit/arr_Prosit_2023_intensity_XL_CMS2_seq_2.npy", array
# )
"""
