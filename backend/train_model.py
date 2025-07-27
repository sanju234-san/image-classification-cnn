import os
import json
import numpy as np
import asyncio
import nest_asyncio
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket

# --- Settings ---
MONGODB_URL = "mongodb://localhost:27017"
DATABASE_NAME = "image_classifier"
COLLECTION_NAME = "dataset_images"
BUCKET_NAME = "dataset_images"
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10

nest_asyncio.apply()  # patch asyncio loop in Jupyter/Anaconda environments

# --- MongoDB Data Generator ---
class MongoDBDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, client, db_name, collection_name, bucket_name,
                 batch_size=32, img_size=(150, 150), validation_split=0.2, subset='training', **kwargs):
        super().__init__(**kwargs)
        self.client = client
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.fs_bucket = AsyncIOMotorGridFSBucket(self.db, bucket_name=bucket_name)
        self.batch_size = batch_size
        self.img_size = img_size
        self.subset = subset
        self.loop = asyncio.get_event_loop()

        self.images_data = self.loop.run_until_complete(self._load_dataset_metadata())
        self.class_names = sorted(list(set([img['class'] for img in self.images_data])))
        self.num_classes = len(self.class_names)
        self.class_indices = {name: idx for idx, name in enumerate(self.class_names)}

        total = len(self.images_data)
        val_size = int(total * validation_split)
        if self.subset == 'training':
            self.images_data = self.images_data[val_size:]
        else:
            self.images_data = self.images_data[:val_size]

        self.indexes = np.arange(len(self.images_data))
        np.random.shuffle(self.indexes)

    async def _load_dataset_metadata(self):
        cursor = self.collection.find({})
        images = []
        async for doc in cursor:
            images.append(doc)
        print(f"✅ Loaded {len(images)} documents from MongoDB")
        return images

    def __len__(self):
        return len(self.images_data) // self.batch_size

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_docs = [self.images_data[k] for k in batch_indexes]
        return self.loop.run_until_complete(self._load_batch(batch_docs))

    async def _load_batch(self, batch_docs):
        X = np.empty((self.batch_size, *self.img_size, 3))
        y = np.empty((self.batch_size, self.num_classes))
        for i, doc in enumerate(batch_docs):
            try:
                grid_out = await self.fs_bucket.open_download_stream(doc['file_id'])
                image_data = await grid_out.read()

                img = Image.open(io.BytesIO(image_data)).convert("RGB")
                img = img.resize(self.img_size)
                img_array = np.array(img) / 255.0
                X[i] = img_array

                class_idx = self.class_indices[doc['class']]
                y[i] = tf.keras.utils.to_categorical(class_idx, self.num_classes)
            except Exception as e:
                print(f"⚠️ Failed to load image: {e}")
                X[i] = np.zeros((*self.img_size, 3))
                y[i] = np.zeros(self.num_classes)
        return X, y

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

# --- Main Training Logic ---
async def setup_generators():
    client = AsyncIOMotorClient(MONGODB_URL)
    await client.admin.command('ping')
    print("✅ Connected to MongoDB")
    train_gen = MongoDBDataGenerator(client, DATABASE_NAME, COLLECTION_NAME, BUCKET_NAME,
                                     batch_size=BATCH_SIZE, img_size=IMG_SIZE, subset='training')
    val_gen = MongoDBDataGenerator(client, DATABASE_NAME, COLLECTION_NAME, BUCKET_NAME,
                                   batch_size=BATCH_SIZE, img_size=IMG_SIZE, subset='validation')
    return train_gen, val_gen, client

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

async def main():
    train_gen, val_gen, client = await setup_generators()

    if len(train_gen) == 0:
        print("❌ No training data found. Exiting.")
        return

    model = build_model((*IMG_SIZE, 3), train_gen.num_classes)

    print("🚀 Training model...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        verbose=1
    )

    # --- Save Model ---
    os.makedirs("models", exist_ok=True)
    model.save("models/model.h5")
    print("✅ Model saved to models/model.h5")

    with open("models/class_names.json", "w") as f:
        json.dump(train_gen.class_names, f)
    print(f"✅ Class names saved: {train_gen.class_names}")

    training_stats = {
        "epochs": EPOCHS,
        "training_samples": len(train_gen) * BATCH_SIZE,
        "validation_samples": len(val_gen) * BATCH_SIZE,
        "classes": train_gen.class_names,
        "final_accuracy": float(history.history['accuracy'][-1]),
        "final_val_accuracy": float(history.history['val_accuracy'][-1]),
    }
    with open("models/training_stats.json", "w") as f:
        json.dump(training_stats, f, indent=2)
    print("✅ Training stats saved")

    client.close()

# --- Run Script ---
if __name__ == "__main__":
    asyncio.run(main())
