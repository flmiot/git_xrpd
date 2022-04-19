import re
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

from mpl_toolkits.mplot3d import Axes3D

class Pilatus2ThetaScan():
    PIXEL_SIZE      = 172e-3
    CHIP_Y          = 487
    CHIP_X          = 195
    SUPER_SAMPLE    = 1

    def __init__(self, images, fio, parameters):
        self.img = images
        self.i0, self.tt = self.read_fio(fio)
        self.p = parameters

    def read_fio(self, fio_path):
        with open(fio_path, 'r') as content_file:
            content = content_file.read()

        pattern = r'\s*([+-]*\d+\.*\d*[e0-9-+]*)\s' * 17
        matches = re.findall(pattern, content)

        i0 = np.empty(len(matches))
        tt = np.empty(len(matches))

        for index, match in enumerate(matches):
            i0[index] = match[3]
            tt[index] = match[0]

        return i0, tt

    def get_diffractogram(self):
        x0, x1 = self.p['pil_pixel_x0'], self.p['pil_pixel_x1']
        y0, y1 = self.p['pil_pixel_y0'], self.p['pil_pixel_y1']


        pixel_positions_x = np.arange(
            -1*self.p['pil_pixel_direct_beam_x']+1,
            self.CHIP_X - self.p['pil_pixel_direct_beam_x']+1
            ) * self.PIXEL_SIZE
        pixel_positions_y = np.arange(
            -1*self.p['pil_pixel_direct_beam_y']+1,
            self.CHIP_Y - self.p['pil_pixel_direct_beam_y']+1
            ) * self.PIXEL_SIZE

        # XX, YY = np.meshgrid(pixel_positions_x, pixel_positions_y)

        phi = np.arctan(pixel_positions_y / self.p['pil_distance'])
        # correction because of flat detector surface
        phi[phi > 0] = phi[phi > 0] * phi[phi > 0]/np.tan(phi[phi > 0])
        phi[phi < 0] = phi[phi < 0] * phi[phi < 0]/np.tan(phi[phi < 0])

        # Convert to degree
        phi = phi*180/np.pi

        l = np.ptp(self.tt) / (180/np.pi*np.arctan(self.PIXEL_SIZE / self.p['pil_distance']))
        bins = int(round(l)) * self.SUPER_SAMPLE
        min_a, max_a = np.min(np.min(self.tt) + phi),  np.max(np.max(self.tt) + phi)
        diffrgm_x, diffrgm_y = np.linspace(min_a, max_a, bins), np.zeros(bins)
        intensity_mask = np.zeros(bins) + 1e-6

        images = self.img / self.i0[:, None, None]

        plt.plot(self.i0)
        plt.show()


        for idx, image in enumerate(images[:, x0:x1+1, y0:y1+1]):

            # position in space of center pixel
            cpx = 0
            cpy = np.sin(self.tt[idx] * np.pi / 180) * self.p['pil_distance']
            cpz = self.p['pil_distance'] * np.cos(self.tt[idx] * np.pi / 180)

            # positions of all other pixels
            px = pixel_positions_x + cpx
            py = pixel_positions_y * np.cos(self.tt[idx] * np.pi / 180) + cpy
            pz = cpz - pixel_positions_y * np.sin(self.tt[idx] * np.pi / 180)

            XX, YY = np.meshgrid(px, py)
            ZZ = np.tile(pz, (px.shape[0], 1)).T
            #
            # plt.plot(cpz, cpy , '.')
            # plt.xlim([0, 1100])
            # plt.ylim([0, 1100])
            # print(cpz, cpy)


            pixel_source_distance = np.sqrt(XX**2+YY**2+ZZ**2)

            # plt.imshow(pixel_source_distance)
            # plt.colorbar()
            # plt.show()

            perp_to_beam = np.sqrt(XX**2+YY**2)
            TT = (np.arcsin(perp_to_beam / pixel_source_distance) * 180 / np.pi).T
            TT = TT[x0:x1+1, y0:y1+1]

            l = list(range(TT[y0:y1+1].shape[1]))


            for idr, x, y in zip(l, TT, image):
                f = interp.interp1d(x, y, 'nearest', fill_value = 0, bounds_error = False)
                m = interp.interp1d(x, np.ones(len(x)), fill_value = 0, bounds_error = False)

                diffrgm_y += f(diffrgm_x)
                intensity_mask += m(diffrgm_x)

            # if idx % 5 == 0:
            #     plt.plot(diffrgm_x, diffrgm_y)
            plt.pause(0.01)
            # plt.plot(intensity_mask)
            # plt.show()

            progress = int(idx / self.img.shape[0] * 50)
            fmt = 'Full analysis pending:  [{:<50}]'.format('='*progress)
            print(fmt, end = '\r')
        # plt.show()


        return diffrgm_x, diffrgm_y / intensity_mask

    def get_diffractogram_fast(self):
        x0, x1 = self.p['pil_pixel_x0'], self.p['pil_pixel_x1']
        y0, y1 = self.p['pil_pixel_y0'], self.p['pil_pixel_y1']


        pixel_positions_x = np.arange(
            -1*self.p['pil_pixel_direct_beam_x']+1,
            self.CHIP_X - self.p['pil_pixel_direct_beam_x']+1
            ) * self.PIXEL_SIZE
        pixel_positions_y = np.arange(
            -1*self.p['pil_pixel_direct_beam_y']+1,
            self.CHIP_Y - self.p['pil_pixel_direct_beam_y']+1
            ) * self.PIXEL_SIZE

        # XX, YY = np.meshgrid(pixel_positions_x, pixel_positions_y)

        phi = np.arctan(pixel_positions_y / self.p['pil_distance'])
        # correction because of flat detector surface
        # phi[phi > 0] = phi[phi > 0] * phi[phi > 0]/np.tan(phi[phi > 0])
        # phi[phi < 0] = phi[phi < 0] * phi[phi < 0]/np.tan(phi[phi < 0])

        # Convert to degree
        phi = phi*180/np.pi

        l = np.ptp(self.tt) / (180/np.pi*np.arctan(self.PIXEL_SIZE / self.p['pil_distance']))
        bins = int(round(l)) * self.SUPER_SAMPLE
        min_a, max_a = np.min(np.min(self.tt) + phi),  np.max(np.max(self.tt) + phi)
        diffrgm_x, diffrgm_y = np.linspace(min_a, max_a, bins), np.zeros(bins)

        for idx, image in enumerate(self.img[:, x0:x1+1, y0:y1+1]):
            x = self.tt[idx] + phi[y0:y1+1]
            y = np.sum(image, axis = 0)
            f = interp.interp1d(x, y, 'nearest', fill_value = 0, bounds_error = False)
            diffrgm_y += f(diffrgm_x)

        #
        #     if idx % 5 == 0:
        #         plt.plot(x, y)
        # plt.show()


        return diffrgm_x, diffrgm_y
