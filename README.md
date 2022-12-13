```python
class Data(Dataset):
    def __init__(self,
                 root_dir,
                 train=True,
                 fold=1,
                 rotate=40,
                 flip=True,
                 random_crop=True,
                 scale1=512):

        self.root_dir = root_dir
        self.train = train
        self.fold = fold
        self.rotate = rotate
        self.flip = flip
        self.random_crop = random_crop
        self.transform = transforms.ToTensor()
        self.resize = scale1
        # print(self.fold)
        self.images, self.groundtruth = load_patch(self.root_dir, self.train, self.fold)

    def __len__(self):
        return len(self.images)

    def RandomFlip(self, image, label):
        axes = (0, 1, 2)
        for axis in axes:
            image = np.flip(image, axis)
            label = np.flip(label, axis)
        return image, label

    def RandomRotate(self, image, label):
        axis = (1, 2)
        k = random.randint(0, 4)
        image = np.rot90(image, k, axis)
        label = np.rot90(label, k, axis)
        return image, label

    def RandomContrast(self, input, alpha=(0.5, 1.5), mean=0.0):
        alpha = random.uniform(alpha[0], alpha[1])
        result = mean + alpha * (input - mean)
        return np.clip(result, -1, 1)

    def RandomCrop(self, image, label, crop_factor=(0, 0, 0)):
        """
        Make a random crop of the whole volume
        :param image:
        :param label:
        :param crop_factor: The crop size that you want to crop
        :return:
        """
        w, h, d = image.shape
        # print(w, crop_factor[0], h, crop_factor[1], d, crop_factor[2])
        z = random.randint(0, w - crop_factor[0])
        y = random.randint(0, h - crop_factor[1])
        x = random.randint(0, d - crop_factor[2])

        image = image[z:z + crop_factor[0], y:y + crop_factor[1], x:x + crop_factor[2]]
        label = label[z:z + crop_factor[0], y:y + crop_factor[1], x:x + crop_factor[2]]
        return image, label

    def __getitem__(self, idx):
        img_path = self.images[idx]
        gt_path = self.groundtruth[idx]

        image = sitk.ReadImage(img_path)
        ima = sitk.GetArrayFromImage(image).astype(np.float32)  # [x,y,z] -> [z,y,x]

        label = sitk.ReadImage(gt_path)
        # if use CE loss, type: astype(np.int64), or use MSE type: astype(np.float32)
        lab = sitk.GetArrayFromImage(label).astype(np.int64)  # [x,y,z] -> [z,y,x]

        image, label = self.RandomCrop(ima, lab, crop_factor=(64, 128, 256))  # [z,y,x]
        # image = standardization_intensity_normalization(image, 'float32')
        while (image.any() == 0):
            image, label = self.RandomCrop(ima, lab, crop_factor=(64, 128, 256))  # [z,y,x]
        image = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
        label = torch.from_numpy(np.ascontiguousarray(label)).unsqueeze(0)

        label = label / 255

        return image, label
   ```
