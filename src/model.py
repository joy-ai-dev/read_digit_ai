import numpy as np
from tensorflow.keras.datasets import mnist

# 1. Load and preprocess data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0  # Flatten & normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

# One-hot encode labels
y_train_oh = np.eye(10)[y_train]
y_test_oh = np.eye(10)[y_test]

# 2. Initialize weights
input_size = 784   # 28x28 pixels
hidden_size = 64
output_size = 10

# Weight dimension: previous layer X current layer
# Bias dimesion: 1 x current later
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# 3. Activation functions
def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# 4. Training loop

# lr is Learning rate: Rate at which we shift weight. Too large means model keeps jumping around
# Too small means model will be too slow to learn. 
# 0.1 is right for the toy AI. When I tested, 0.4 gave better results
lr = 0.1

# epochs is the number of time model is trained in the training data. 
epochs = 5

# How many sample images we train the AI before updating weights
# 64 images -> Forward pass -> Compute loss -> Backward pass -> Update weights
batch_size = 64

for epoch in range(epochs):
    for i in range(0, X_train.shape[0], batch_size):
        # Shape of x_batch (64, 784)
        X_batch = X_train[i:i+batch_size]

        # Shape of y_batch (64, 10)
        y_batch = y_train_oh[i:i+batch_size]

        # Forward pass
        # Shape: (64 x 784) . (784 x 64) + broadcast(1 x 64) = 64 x 64
        z1 = X_batch @ W1 + b1
        a1 = relu(z1)
        
        # Shape: (64 x 64) . (64 x 10) + broadcast (1 x 10) = 64 x 10 
        z2 = a1 @ W2 + b2
        a2 = softmax(z2)

        # Loss (cross-entropy)
        # loss is how much probability varied from the actual result 1 
        # if loss is small, model is confident. If loss is large, it is wrog or unsure
        loss = -np.mean(np.sum(y_batch * np.log(a2 + 1e-8), axis=1))

        # Backward pass
        dz2 = a2 - y_batch
        dW2 = a1.T @ dz2 / batch_size
        db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size

        dz1 = (dz2 @ W2.T) * relu_deriv(z1)
        dW1 = X_batch.T @ dz1 / batch_size
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size

        # Update weights
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# 5. Prediction
z1 = X_test @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
preds = np.argmax(softmax(z2), axis=1)

accuracy = np.mean(preds == y_test)

print(f"Test Accuracy: {accuracy:.4f}")

def draw_and_predict():
    import tkinter as tk
    from PIL import Image, ImageGrab
    import numpy as np

    # --- Helper: predict digit ---
def draw_and_predict():
    import tkinter as tk
    from tkinter import messagebox
    from PIL import Image, ImageDraw, ImageTk
    import numpy as np

    # --- helper: preprocess PIL image into MNIST format ---
    def preprocess_pil_image_for_mnist(pil_img):
        """Return (1,784) array and 28x28 PIL preview image (exactly what goes to model)."""
        img = pil_img.convert('L')        # grayscale
        arr = 255 - np.array(img)         # invert (black digit -> white digit like MNIST)

        # find bounding box of drawn digit
        coords = np.column_stack(np.where(arr > 10))
        if coords.size:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            crop = arr[y0:y1+1, x0:x1+1]
            h, w = crop.shape

            # scale so largest side = 20
            scale = 20.0 / max(h, w)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            crop_img = Image.fromarray(crop).resize((new_w, new_h), Image.Resampling.LANCZOS)

            # paste into 28x28 center
            new_img = Image.new('L', (28, 28), color=0)
            paste_x = (28 - new_w) // 2
            paste_y = (28 - new_h) // 2
            new_img.paste(crop_img, (paste_x, paste_y))
        else:
            new_img = img.resize((28, 28), Image.Resampling.LANCZOS)

        arr = np.array(new_img).astype(np.float32) / 255.0
        return arr.reshape(1, -1), new_img

    # --- tkinter setup ---
    root = tk.Tk()
    root.title("Draw a Digit")

    canvas = tk.Canvas(root, width=280, height=280, bg="white")
    canvas.pack()

    pil_image = Image.new("L", (280, 280), "white")
    draw = ImageDraw.Draw(pil_image)

    preview_label = tk.Label(root)
    preview_label.pack()

    def paint(event):
        x1, y1 = (event.x - 4), (event.y - 4)
        x2, y2 = (event.x + 4), (event.y + 4)
        canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        draw.ellipse([x1, y1, x2, y2], fill="black")

    canvas.bind("<B1-Motion>", paint)

    def clear_canvas():
        canvas.delete("all")
        draw.rectangle([0, 0, 280, 280], fill="white")
        preview_label.config(image="")
        preview_label.image = None

    def on_predict():
        # preprocess drawn image
        x, preview_img = preprocess_pil_image_for_mnist(pil_image)

        # forward pass through trained model
        z1 = x @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        probs = softmax(z2)[0]
        pred = int(np.argmax(probs))

        # show preview (scaled for visibility)
        preview_img = preview_img.resize((140, 140), Image.Resampling.NEAREST)
        preview_tk = ImageTk.PhotoImage(preview_img)  # ✅ correct way
        preview_label.config(image=preview_tk)
        preview_label.image = preview_tk  # keep reference

        # debug prints
        print("Input arr shape:", x.shape, "min:", x.min(), "max:", x.max(), "mean:", x.mean())
        top3 = np.argsort(probs)[-3:][::-1]
        print("Prediction:", pred, "Top3:", top3, "Probs:", np.round(probs[top3], 3))

        messagebox.showinfo("Prediction", f"Predicted digit: {pred}")
        clear_canvas()  # Automatically clear after showing result
    btn_frame = tk.Frame(root)
    btn_frame.pack()
    tk.Button(btn_frame, text="Predict", command=on_predict).pack(side="left")
    tk.Button(btn_frame, text="Clear", command=clear_canvas).pack(side="left")

    root.mainloop()





draw_and_predict()