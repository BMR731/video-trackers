import numpy as np
import cv2 as cv
import os

#### Hyper Parameters
EPS = 1e-5
NUM_PRETRAIN = 128
LR = 0.125


#####################


## Data augmention during training , here just use 'Rotation'
def random_wrap(img):
    h, w = img.shape[:2]
    angle = (np.random.rand() - 0.5) * 60
    transform_matrix = cv.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    out = cv.warpAffine(img, transform_matrix, (w, h), borderMode=cv.BORDER_REFLECT)
    return out

## Frequency domain division
def divSpectrums(A, B):
    Ar, Ai = A[..., 0], A[..., 1]
    Br, Bi = B[..., 0], B[..., 1]
    C = (Ar + 1j * Ai) / (Br + 1j * Bi)
    C = np.dstack([np.real(C), np.imag(C)]).copy()
    return C


class MOSSE:
    def __init__(self, frame, rect):
        x, y, w, h = rect
        x1, y1, x2, y2 = x, y, x + w, y + h
        w, h = map(cv.getOptimalDFTSize, [x2 - x1, y2 - y1]) # for get optimal size for FFT
        x1, y1 = (x1 + x2 - w) // 2, (y1 + y2 - h) // 2
        self.pos = x, y = x1 + 0.5 * (w - 1), y1 + 0.5 * (h - 1)
        self.size = w, h
        img = cv.getRectSubPix(frame, (w, h), (x, y))

        self.win = cv.createHanningWindow((w, h), cv.CV_32F)
        g = np.zeros((h, w), np.float32)
        g[h // 2, w // 2] = 1
        g = cv.GaussianBlur(g, (-1, -1), 2.0)
        g /= g.max()

        self.G = cv.dft(g, flags=cv.DFT_COMPLEX_OUTPUT)
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)
        for i in range(NUM_PRETRAIN):
            a = self.preprocess(random_wrap(img))
            A = cv.dft(a, flags=cv.DFT_COMPLEX_OUTPUT)
            self.H1 += cv.mulSpectrums(self.G, A, 0, conjB=True)
            self.H2 += cv.mulSpectrums(A, A, 0, conjB=True)
        self.update_kernal()
        self.update(frame)

    def preprocess(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = np.log(np.float32(img) + 1.0)
        img = (img - img.mean(0)) / (img.std() + EPS)
        return img * self.win

    def update_kernal(self):
        self.H = divSpectrums(self.H1, self.H2)
        self.H[..., 1] *= -1

    # use correlation to get response map and find max area
    # but notice here, the img use last rect, not origin frame
    # so if the target move so fast, we may loss .
    def correlate(self, img):
        C = cv.mulSpectrums(cv.dft(img, flags=cv.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)
        resp = cv.idft(C, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
        h, w = resp.shape
        # get max resp loc
        _, mval, _, (mx, my) = cv.minMaxLoc(resp)
        side_resp = resp.copy()
        cv.rectangle(side_resp, (mx - 5, my - 5), (mx + 5, my + 5), 0, -1)
        cv.imshow('resp',side_resp)
        cv.waitKey(100)
        smean, sstd = side_resp.mean(), side_resp.std()
        psr = (mval - smean) / (sstd + EPS)
        return resp, (mx - w // 2, my - h // 2), psr

    def update(self, frame, lr=LR):
        (x, y), (w, h) = self.pos, self.size
        self.last_img = img = cv.getRectSubPix(frame, (w, h), (x, y))
        img = self.preprocess(img)
        self.last_resp, (dx, dy), self.psr = self.correlate(img)
        self.good = self.psr > 8.0
        if not self.good:
            self.visualize(frame)
            return
        self.pos = x + dx, y + dy
        self.last_img = img = cv.getRectSubPix(frame, (w, h), self.pos)
        img = self.preprocess(img)
        ## online update
        A = cv.dft(img, flags=cv.DFT_COMPLEX_OUTPUT)
        H1 = cv.mulSpectrums(self.G, A, 0, conjB=True)
        H2 = cv.mulSpectrums(A, A, 0, conjB=True)
        self.H1 = self.H1 * (1 - lr) + H1 * lr
        self.H2 = self.H2 * (1 - lr) + H2 * lr
        self.visualize(frame)
        self.update_kernal()

    def visualize(self, vis):
        (x, y), (w, h) = self.pos, self.size
        x1, y1, x2, y2 = int(x - 0.5 * w), int(y - 0.5 * h), int(x + 0.5 * w), int(y + 0.5 * h)
        cv.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))
        if self.good:
            cv.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
        else:
            cv.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
            cv.line(vis, (x2, y1), (x1, y2), (0, 0, 255))
        cv.putText(vis, 'PSR: %.2f' % self.psr, (x1 + 1, y2 + 17), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2,
                   lineType=cv.LINE_AA)
        cv.putText(vis, 'PSR: %.2f' % self.psr, (x1, y2 + 16), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255),
                   lineType=cv.LINE_AA)
        cv.imshow('frame', vis)
        cv.waitKey(100)


if __name__ == '__main__':
    video_path = 'surfer/'
    frames_list = []
    for frame in os.listdir(video_path):
        if os.path.splitext(frame)[1] == '.jpg':
            frames_list.append(os.path.join(video_path, frame))
    frames_list.sort()
    tracker = 1
    for idx in range(len(frames_list)):
        img = cv.imread(frames_list[idx])
        if idx == 0:
            init_rect = cv.selectROI('Target', img, showCrosshair=False, fromCenter=False)
            tracker = MOSSE(img, init_rect)
        else:
            tracker.update(img)
