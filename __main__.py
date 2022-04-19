import os
import numpy as np

from diffraction import Pilatus2ThetaScan
from tools import read_images
import matplotlib.pyplot as plt

plt.ion()

output = 'C:/Users/otteflor/Google Drive/BL9_DELTA_Feb20_XRPD/run04_20_xrd_xfel/reduced'

parameters = {
    'pil_pixel_x0':68 ,
    'pil_pixel_x1':128 ,
    'pil_pixel_y0':40,
    'pil_pixel_y1':487,
    'pil_pixel_direct_beam_x':104,
    'pil_pixel_direct_beam_y':257,
    'pil_distance': 1015
}
run_nr = 16

images = read_images(
    'C:/Users/otteflor/Google Drive/BL9_DELTA_Feb20_XRPD/run04_20_xrd_xfel/pilatus',
    '{:0>5}'.format(run_nr)
    )
fio = 'C:/Users/otteflor/Google Drive/BL9_DELTA_Feb20_XRPD/run04_20_xrd_xfel/run04_20_{:0>5}.FIO'.format(run_nr)

tts = Pilatus2ThetaScan(images, fio, parameters)
tt, intensity = tts.get_diffractogram_fast()
plt.plot(tt, intensity / np.max(intensity), label = 'fast analysis')
plt.show(block = False)
plt.pause(0.1)
tt, intensity = tts.get_diffractogram()
plt.plot(tt, intensity / np.max(intensity), label = 'full analysis')
plt.legend()
plt.ioff()
path = os.path.join(output, 'run_reduced_full_{:0>5}.dat'.format(run_nr))
np.savetxt(path, np.array([tt, intensity]).T)

plt.show()
