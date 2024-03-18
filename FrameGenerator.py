import pathlib
import random
import tensorflow as tf
import numpy as np
from preprocessing import frames_from_video_file

class FrameGenerator:
  def __init__(self, split, path = None,n_frames = 20, training = False):
    """ Returns a set of frames with their associated label. 

      Args:
        path: Video file paths. A pathlib path
        n_frames: Number of frames. 
        training: Boolean to determine if training dataset is being created.
    """
    if path is not None:
      self.path = path
    else:
      self.path = pathlib.Path("./data")
    self.split = split
    self.n_frames = n_frames
    self.training = training
    self.class_names = sorted(set(p.name for p in (self.path / "train").iterdir() if p.is_dir()))
    self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

  def get_files_and_class_names(self):
    if self.split == "train":
        pattern = "*/*/*.mp4"
        video_paths = list(self.path.glob(pattern))
        classes = [p.parent.name for p in video_paths] 
    else:
        pattern = "*/*.mp4"
        video_paths = list(self.path.glob(pattern))
        txt = open(self.path / "validate.txt", "r")
        map = dict()
        for line in txt:
            arr = line.strip().split('\t')
            map[arr[2]] = int(arr[1])
        class_enum = ['Jump', 'Run', 'Sit', 'Stand', 'Turn', 'Walk']
        classes = [class_enum[map[p.name]] for p in video_paths]
    
    return video_paths, classes

  def __call__(self):
    video_paths, classes = self.get_files_and_class_names()

    pairs = list(zip(video_paths, classes))

    if self.training:
      random.shuffle(pairs)

    for path, name in pairs:
      video_frames = frames_from_video_file(path, self.n_frames) 
      label = self.class_ids_for_name[name] # Encode labels
      yield video_frames, label

# input shape (before batching) [frames, width, height, channels]
# output: predicted class,a single int 0, ..., 4, 5
output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))

# train_fg = FrameGenerator(split='train', n_frames=20, training=True)
# val_fg = FrameGenerator(split='validate', n_frames=20)

train_ds = tf.data.Dataset.from_generator(FrameGenerator(split='train', n_frames=20, training=True), output_signature=output_signature)
val_ds = tf.data.Dataset.from_generator(FrameGenerator(split='validate', n_frames=20), output_signature=output_signature)

'''
# To check if the dataset if shuffled
for frames, labels in train_ds.take(10):
  print(labels)
                               

  

# Print the shapes of the data
train_frames, train_labels = next(iter(train_ds))
print(f'Shape of training set of frames: {train_frames.shape}')
print(f'Shape of training labels: {train_labels.shape}')

val_frames, val_labels = next(iter(val_ds))
print(f'Shape of validation set of frames: {val_frames.shape}')
print(f'Shape of validation labels: {val_labels.shape}')

'''         

AUTOTUNE = tf.data.AUTOTUNE 

train_ds = train_ds.cache().shuffle(5).prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().shuffle(5).prefetch(buffer_size = AUTOTUNE)

train_ds = train_ds.batch(2)
val_ds = val_ds.batch(2)

# train_frames, train_labels = next(iter(train_ds))
# print(f'Shape of training set of frames: {train_frames.shape}')
# print(f'Shape of training labels: {train_labels.shape}')

# val_frames, val_labels = next(iter(val_ds))
# print(f'Shape of validation set of frames: {val_frames.shape}')
# print(f'Shape of validation labels: {val_labels.shape}')



e2e = True

if e2e:
    # End-to-end train
    
    net = tf.keras.applications.EfficientNetB0(include_top = False)
    net.trainable = False
    
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(scale=255),
        tf.keras.layers.TimeDistributed(net),
        tf.keras.layers.Dense(10),
        tf.keras.layers.GlobalAveragePooling3D()
    ])
    
    model.compile(optimizer = 'adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                metrics=['accuracy'])

    model.fit(train_ds, 
            epochs = 20,
            validation_data = val_ds,
            callbacks = tf.keras.callbacks.EarlyStopping(patience = 10, monitor = 'val_loss')) 

else:
    ENB0 = tf.keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False
    )
    
    def get_features_and_labels(dataset):
        all_features = []
        all_labels = []
        
        k = 0
        for frames, label in dataset:
            k = k+1
            features = ENB0.predict(frames)
            all_features.append(features)
            all_labels.append(label)
        print('Extracted features from {} samples', format(k))
        return np.array(all_features), np.concatenate([np.expand_dims(i,axis=0) for i in all_labels])

    FLAG = True
    if FLAG:
        all_features = np.load("all_features.npy")
        all_labels = np.load("all_labels.npy")
        val_features = np.load("val_all_features.npy")
        val_labels = np.load("val_all_labels.npy")
        
        
    else:
        all_features, all_labels = get_features_and_labels(train_ds)
        val_features, val_labels = get_features_and_labels(val_ds)
        np.save("val_all_features.npy", val_features)
        np.save("val_all_labels.npy", val_labels)
    
    from sklearn import metrics, svm
    
    svc = svm.SVC(gamma="auto")
    train_features_flatten = np.array([np.ndarray.flatten(x) for x in all_features])
    val_features_flatten = np.array([np.ndarray.flatten(x) for x in val_features])
    svc.fit(train_features_flatten, all_labels)
    
    train_acc = svc.score(train_features_flatten, all_labels)
    val_acc = svc.score(val_features_flatten, val_labels)
    
    print("accuracy on training set and validating set is {0} and {1}, respectively".format(train_acc, val_acc))
    
    
    #print('[CONCATENATED]All features shape:{0}, all labels shape: {1}'.format(all_features.shape, all_labels.shape))