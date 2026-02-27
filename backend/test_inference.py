from PIL import Image
from model import predict
import time

image = Image.open("test.jpg")

start = time.time()
result = predict(image)
end = time.time()

print(result)
print("Inference Time:", end - start, "seconds")
