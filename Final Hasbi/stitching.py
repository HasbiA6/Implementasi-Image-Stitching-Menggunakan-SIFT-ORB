import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


# ==============================
# LOAD IMAGE
# ==============================
img1 = cv2.imread("image1.jpg")
img2 = cv2.imread("image2.jpg")

if img1 is None or img2 is None:
    print("Error: image not found!")
    exit()

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# ==============================
# FEATURE DETECTION
# ==============================
def detect_and_compute(method, gray):
    if method == "SIFT":
        detector = cv2.SIFT_create()
    elif method == "ORB":
        detector = cv2.ORB_create(nfeatures=2000)
    else:
        raise ValueError("Unsupported method")

    kp, desc = detector.detectAndCompute(gray, None)
    return kp, desc


# ==============================
# FEATURE MATCHING
# ==============================
def match_features(method, desc1, desc2):

    if method == "ORB":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2)

    matches = matcher.knnMatch(desc1, desc2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:   # lebih ketat
            good.append(m)

    return good


# ==============================
# STITCH FUNCTION (FIXED)
# ==============================
def stitch(method):

    print(f"\n===== {method} =====")

    start = time.time()

    kp1, desc1 = detect_and_compute(method, gray1)
    kp2, desc2 = detect_and_compute(method, gray2)

    good_matches = match_features(method, desc1, desc2)

    print("Keypoints img1:", len(kp1))
    print("Keypoints img2:", len(kp2))
    print("Good matches:", len(good_matches))

    if len(good_matches) < 10:
        print("Not enough matches!")
        return

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Cari homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # ===== AUTO CANVAS SIZE FIX =====
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    pts_img1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts_img1_transformed = cv2.perspectiveTransform(pts_img1, H)

    pts = np.concatenate((pts_img1_transformed,
                          np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)), axis=0)

    [xmin, ymin] = np.int32(pts.min(axis=0).ravel())
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel())

    translation_dist = [-xmin, -ymin]

    H_translation = np.array([[1,0,translation_dist[0]],
                              [0,1,translation_dist[1]],
                              [0,0,1]])

    result = cv2.warpPerspective(img1,
                                 H_translation.dot(H),
                                 (xmax-xmin, ymax-ymin))

    result[translation_dist[1]:h2+translation_dist[1],
           translation_dist[0]:w2+translation_dist[0]] = img2

    end = time.time()

    print("Execution time:", round(end-start,4), "seconds")

    # DRAW MATCHES
    match_img = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # SHOW RESULT
    plt.figure(figsize=(15,5))
    plt.title(f"{method} - Feature Matching")
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    plt.figure(figsize=(15,5))
    plt.title(f"{method} - Panorama Result")
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


# ==============================
# RUN METHODS
# ==============================
stitch("SIFT")
stitch("ORB")
