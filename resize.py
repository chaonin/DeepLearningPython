from PIL import Image  
  
def resize_image(input_image_path, output_image_path, size):  
    original_image = Image.open(input_image_path)  
    resized_image = original_image.resize(size)  
    resized_image.save(output_image_path)  
  
# 使用函数将图片压缩为28x28像素  
resize_image('moon.jpeg', 'output_28x28.jpeg', (28, 28))
