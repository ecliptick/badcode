import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy, scipy.misc, scipy.signal
import time
from skimage import exposure as ex
import torch
import torch.nn as nn

# IB=is bright
def isbright(image, dim=10, thresh=0.35):
    # Resize image to 10x10
    image = cv2.resize(image, (dim, dim))
    # Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L / np.max(L)
    # Return True if mean is greater than thresh else False
    return np.mean(L) > thresh


def he(img):
    if len(img.shape) == 2:  # gray
        outImg = ex.equalize_hist(img[:, :]) * 255
    elif len(img.shape) == 3:  # RGB
        outImg = np.zeros((img.shape[0], img.shape[1], 3))
        for channel in range(img.shape[2]):
            outImg[:, :, channel] = ex.equalize_hist(img[:, :, channel]) * 255

    outImg[outImg > 255] = 255
    outImg[outImg < 0] = 0
    return outImg.astype(np.uint8)


def heimg(img_name):
    img = cv2.imread(img_name)
    result = he(img)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()


def build_is_hist(img):
    hei = img.shape[0]
    wid = img.shape[1]
    ch = img.shape[2]
    Img = np.zeros((hei + 4, wid + 4, ch))
    for i in range(ch):
        Img[:, :, i] = np.pad(img[:, :, i], (2, 2), "edge")
    hsv = matplotlib.colors.rgb_to_hsv(Img)
    hsv[:, :, 0] = hsv[:, :, 0] * 255
    hsv[:, :, 1] = hsv[:, :, 1] * 255
    hsv[hsv > 255] = 255
    hsv[hsv < 0] = 0
    hsv = hsv.astype(np.uint8).astype(np.float64)
    fh = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    fv = fh.conj().T

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    I = hsv[:, :, 2]

    dIh = scipy.signal.convolve2d(I, np.rot90(fh, 2), mode="same")
    dIv = scipy.signal.convolve2d(I, np.rot90(fv, 2), mode="same")
    dIh[dIh == 0] = 0.00001
    dIv[dIv == 0] = 0.00001
    dI = np.sqrt(dIh**2 + dIv**2).astype(np.uint32)
    di = dI[2 : hei + 2, 2 : wid + 2]

    dSh = scipy.signal.convolve2d(S, np.rot90(fh, 2), mode="same")
    dSv = scipy.signal.convolve2d(S, np.rot90(fv, 2), mode="same")
    dSh[dSh == 0] = 0.00001
    dSv[dSv == 0] = 0.00001
    dS = np.sqrt(dSh**2 + dSv**2).astype(np.uint32)
    ds = dS[2 : hei + 2, 2 : wid + 2]

    h = H[2 : hei + 2, 2 : wid + 2]
    s = S[2 : hei + 2, 2 : wid + 2]
    i = I[2 : hei + 2, 2 : wid + 2].astype(np.uint8)

    Imean = scipy.signal.convolve2d(I, np.ones((5, 5)) / 25, mode="same")
    Smean = scipy.signal.convolve2d(S, np.ones((5, 5)) / 25, mode="same")

    Rho = np.zeros((hei + 4, wid + 4))
    for p in range(2, hei + 2):
        for q in range(2, wid + 2):
            tmpi = I[p - 2 : p + 3, q - 2 : q + 3]
            tmps = S[p - 2 : p + 3, q - 2 : q + 3]
            corre = np.corrcoef(tmpi.flatten("F"), tmps.flatten("F"))
            Rho[p, q] = corre[0, 1]

    rho = np.abs(Rho[2 : hei + 2, 2 : wid + 2])
    rho[np.isnan(rho)] = 0
    rd = (rho * ds).astype(np.uint32)
    Hist_I = np.zeros((256, 1))
    Hist_S = np.zeros((256, 1))

    for n in range(0, 255):
        temp = np.zeros(di.shape)
        temp[i == n] = di[i == n]
        Hist_I[n + 1] = np.sum(temp.flatten("F"))
        temp = np.zeros(di.shape)
        temp[i == n] = rd[i == n]
        Hist_S[n + 1] = np.sum(temp.flatten("F"))

    return Hist_I, Hist_S


def dhe(img, alpha=0.5):
    hist_i, hist_s = build_is_hist(img)
    hist_c = alpha * hist_s + (1 - alpha) * hist_i
    hist_sum = np.sum(hist_c)
    hist_cum = hist_c.cumsum(axis=0)

    hsv = matplotlib.colors.rgb_to_hsv(img)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    i = hsv[:, :, 2].astype(np.uint8)

    c = hist_cum / hist_sum
    s_r = c * 255
    i_s = np.zeros(i.shape)
    for n in range(0, 255):
        i_s[i == n] = s_r[n + 1] / 255.0
    i_s[i == 255] = 1
    hsi_o = np.stack((h, s, i_s), axis=2)
    result = matplotlib.colors.hsv_to_rgb(hsi_o)

    result = result * 255
    result[result > 255] = 255
    result[result < 0] = 0
    return result.astype(np.uint8)


class DCENet(nn.Module):
    """https://li-chongyi.github.io/Proj_Zero-DCE.html"""

    def __init__(self, n=8, return_results=[4, 6, 8]):
        """
        Args
        --------
          n: number of iterations of LE(x) = LE(x) + alpha * LE(x) * (1-LE(x)).
          return_results: [4, 8] => return the 4-th and 8-th iteration results.
        """
        super().__init__()
        self.n = n
        self.ret = return_results

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True)
        self.conv7 = nn.Conv2d(64, 3 * n, kernel_size=3, padding=1, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out1 = self.relu(self.conv1(x))

        out2 = self.relu(self.conv2(out1))
        out3 = self.relu(self.conv3(out2))
        out4 = self.relu(self.conv4(out3))

        out5 = self.relu(self.conv5(torch.cat((out4, out3), 1)))
        out6 = self.relu(self.conv6(torch.cat((out5, out2), 1)))

        alpha_stacked = self.tanh(self.conv7(torch.cat((out6, out1), 1)))

        alphas = torch.split(alpha_stacked, 3, 1)
        results = [x]
        for i in range(self.n):
            # x = x + alphas[i] * (x - x**2)  # as described in the paper
            # sign doesn't really matter becaus of symmetry.
            x = x + alphas[i] * (torch.pow(x, 2) - x)
            if i + 1 in self.ret:
                results.append(x)

        return results, alpha_stacked


def gamma_like(img, enhanced):
    x, y = img.mean(), enhanced.mean()
    gamma = np.log(y) / np.log(x)
    return gamma_correction(img, gamma)


def enh_reader(fp, h, w):
    fp = str(fp)
    img = cv2.imread(fp)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = cv2.resize(img, (w, h))
    return img


def zerodce(image):
    enh_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    device = torch.device("cpu")
    model = DCENet(return_results=[4, 8])
    model.load_state_dict(
        torch.load(
            "models/zero_dce/8LE-color-loss2_best_model.pth",
            map_location=torch.device("cpu"),
        )["model"]
    )
    model.to(device)

    enh_img = torch.from_numpy(np.array(enh_img))
    enh_img = enh_img.float().div(255)
    enh_img = enh_img.permute((2, 0, 1)).contiguous()
    enh_img = enh_img.unsqueeze(0)
    enh_img = enh_img.to(device)
    results, Astack = model(enh_img)
    enhanced_image_base = results[1]

    _, _, h, w = enhanced_image_base.shape

    # this part is re-reading raw image since enhanced_image_base is just to get exposure levels
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = cv2.resize(image, (w, h))

    enh_our = enhanced_image_base[0].permute(1, 2, 0).detach().numpy()
    ori = image
    # print(enh_our)
    # print(ori)
    x, y = ori.mean(), enh_our.mean()
    gamma = np.log(y) / np.log(x)
    return np.power(ori, gamma)

def abright(image):

    cols, rows, _ = image.shape

    brightness = np.sum(image) / (255 * cols * rows)

    minimum_brightness = 1.45

    ratio = brightness / minimum_brightness

    image = cv2.convertScaleAbs(image, alpha = 1 / ratio, beta = 0)

    return image


def perform_image_enhancement(image):
    # this is where we modify to depending on implementation
    # returns array
    if isbright(image):
        to_force = False
        return image, to_force

    else:
        image = zerodce(abright(image))
        image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB)
        image = (image*255).astype(np.uint8)
        to_force = True
        return image, to_force

def perform_abright(image):
    # this is where we modify to depending on implementation
    # returns array
    if isbright(image):
        to_force = False
        return image, to_force

    else:
        image = abright(image)
        to_force = True
        return image, to_force


def perform_image_enhancement_f(image):
    # this is where we modify to depending on implementation
    # returns array
    image = zerodce(abright(image))
    image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB)
    image = (image*255).astype(np.uint8)

    return image

    # image = zerodce(image)
    # image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB)
    # image = (image*255).astype(np.uint8)
    #
    # return image
