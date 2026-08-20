import os

# Suppress TF logs and fix macOS threading conflict with OpenCV
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np

MODEL_PATH = 'handwritten_cnn.keras'

# ─────────────────────────────────────────────
# PHASE 1: Train and save model (if not saved)
# ─────────────────────────────────────────────
def train_and_save():
    import tensorflow as tf

    print("📦 Loading MNIST dataset...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test  = tf.keras.utils.normalize(x_test,  axis=1)

    print("🧠 Building CNN model...")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("🚀 Training CNN (10 epochs) — this runs once and is saved for future use...")
    model.fit(x_train, y_train, epochs=10, verbose=1)

    val_loss, val_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n✅ Training complete! Validation Accuracy: {val_acc*100:.2f}%")

    model.save(MODEL_PATH)
    print(f"💾 Model saved to '{MODEL_PATH}'")


# ─────────────────────────────────────────────
# PHASE 2: Load model and run OpenCV GUI
# ─────────────────────────────────────────────
def run_gui():
    import cv2
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    print(f"\n📂 Loading model from '{MODEL_PATH}'...")
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # Drawing setup
    canvas_size = 500
    canvas = np.zeros((canvas_size, canvas_size, 1), dtype=np.uint8)
    drawing = False
    last_x, last_y = -1, -1

    def draw_doodle(event, x, y, flags, param):
        nonlocal drawing, last_x, last_y

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_x, last_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.line(canvas, (last_x, last_y), (x, y), 255, 30)
                last_x, last_y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            last_x, last_y = -1, -1

        elif event == cv2.EVENT_RBUTTONDOWN:
            predict_and_display()

    def predict_and_display():
        processed_image = canvas.copy()

        _, thresh = cv2.threshold(processed_image, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            cropped_digit = processed_image[y:y+h, x:x+w].squeeze()

            side = max(w, h)
            square_size = max(int(side * 1.2), 28)
            square_image = np.zeros((square_size, square_size), dtype=np.uint8)

            x_offset = (square_size - w) // 2
            y_offset = (square_size - h) // 2
            square_image[y_offset:y_offset+h, x_offset:x_offset+w] = cropped_digit

            image_28x28 = cv2.resize(square_image, (28, 28), interpolation=cv2.INTER_AREA)
            img_normalized = image_28x28.astype('float32') / 255.0
            img_input = np.expand_dims(img_normalized, axis=0)
        else:
            print("\n-------------------------------------------")
            print("❌ Please draw a digit before predicting.")
            print("-------------------------------------------")
            return

        prediction = model.predict(img_input, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        print("\n-------------------------------------------")
        print(f"✅ Prediction Result: The digit is  {predicted_class} ")
        print(f"   Confidence: {confidence:.2f}%")
        print("-------------------------------------------")

        canvas[:] = 0  # Clear canvas after prediction

    # --- Main Loop ---
    cv2.namedWindow('Doodle Predictor')
    cv2.setMouseCallback('Doodle Predictor', draw_doodle)

    print("\n--- 🎨 Doodle Predictor Running ---")
    print("* Left-click and drag  → Draw a digit")
    print("* Right-click          → Predict & clear")
    print("* Press 'c'            → Clear canvas")
    print("* Press 'q' or ESC    → Quit\n")

    while True:
        info_bar = np.zeros((60, canvas_size, 3), dtype=np.uint8)
        cv2.putText(info_bar, "Right-click: Predict  |  C: Clear  |  Q: Quit",
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

        display_img = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        final_display = np.vstack((display_img, info_bar))

        cv2.imshow('Doodle Predictor', final_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('c'):
            canvas[:] = 0

    cv2.destroyAllWindows()
    print("\n👋 Exited Doodle Predictor.")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # Train only if model doesn't exist yet
    if not os.path.exists(MODEL_PATH):
        train_and_save()
    else:
        print(f"✅ Found existing model '{MODEL_PATH}' — skipping training.")

    # Launch GUI after TF is fully done
    run_gui()
