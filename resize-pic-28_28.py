from PIL import Image  

print('input the file you want to resize to 28*28:')
origfile = input()
destfile = origfile + '28_28.jpg'

def resize_image(input_image_path, output_image_path, size):  
    original_image = Image.open(input_image_path)  
    resized_image = original_image.resize(size)  
    resized_image.save(output_image_path)  

def convert_to_black_and_white(image_path, output_path):
    with Image.open(image_path) as image:
        image = image.convert('L')
        image.save(output_path)

def read_pic_pixel(image_path):
    image = Image.open(image_path)
    width, height = image.size
    i = 1 
    for h in range(height):
        for w in range(width):
            pixel_value = image.getpixel((w, h))
            print(str(i)+':'+str(pixel_value))  # 输出像素值，例如：(255, 0, 0)代表红色
            i = i+1
  
# 使用函数将图片压缩为28x28像素  
resize_image(origfile, destfile, (28, 28))
convert_to_black_and_white(destfile, destfile) 
read_pic_pixel(destfile)
