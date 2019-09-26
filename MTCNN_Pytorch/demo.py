import cv2
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector

if __name__ == '__main__':

    pnet, rnet, onet = create_mtcnn_net(
        p_model_path = "./original_model/pnet_epoch.pt",
        r_model_path = "./original_model/rnet_epoch.pt",
        o_model_path = "./original_model/onet_epoch.pt",
        use_cuda = False)
    mtcnn_detector = MtcnnDetector(pnet = pnet, rnet = rnet, onet = onet, min_face_size = 40)

    img = cv2.imread("./test.png")
    bboxs, landmarks = mtcnn_detector.detect_face(img)
    print(bboxs.shape[0])
    for i in range(bboxs.shape[0]):
        bbox = bboxs[i, :4]
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
    cv2.imshow("result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
