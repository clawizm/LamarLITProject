import PIL
from PIL import Image
import os
import time
# doc = aw.Document()
# builder = aw.DocumentBuilder(doc)

# shape = builder.insert_image("Input.png")
# shape.get_shape_renderer().save("Output.png", aw.saving.ImageSaveOptions(aw.SaveFormat.PNG))


def get_all_jpg_files_in_directory(path: str)->list[str]:
    dir_contents = os.listdir(path)
    dir_contents_filtered = [os.path.join(path, file) for file in dir_contents if file.endswith('.jpg')]
    return dir_contents_filtered


def create_and_save_pngs_of_all_files_in_list(jpg_files: list[str])->None:
    for file in jpg_files:
        im = Image.open(file)
        im_resized = im.resize((720, 405))
        im_resized.save(file.replace('jpg', 'png'), "png")
    return

def insertion_sort(lst: list[int])->list[int]:
    for j in range(1, len(lst)):
        val = lst[j]
        i = j-1
        while i >= 0 and val < lst[i]:
            lst[i+1] = lst[i]
            i-=1
        lst[i+1] = val
    return lst

def linear_search(lst: list[int], val: int)->int:
    for i in range(lst):
        if lst[i] == val:
            return i
    return None

def add_binary_numbers(bin_num_one: list[int], bin_num_two: list[int])->list[int]:
    bin_num_sum = []
    i = 0 #bin_num_one pointer
    j = 0 #bin_num_two pointer
    carry = 0
    while i < len(bin_num_one) and j < len(bin_num_two):
        if bin_num_one[i] + bin_num_two[j] == 0:
            carry = 1
        elif bin_num_one[i] + bin_num_two[j] == 1:
            pass
        else:
            bin_num_sum.insert

def selection_sort(lst: list[int])->list[int]:
    for i in range(len(lst)):
        min_idx = i
        for j in range(i+1, len(lst)):
            if lst[j] > lst[min_idx]:
                min_idx = j
        
        lst[min_idx], lst[i] = lst[i], lst[min_idx]
    return lst

def countdown(i):
    print(f'Countdown: {i}')
    if i<=0:
        print('BOOM!')
        return
    else:
        time.sleep(1)
        countdown(i-1)

def factorial(x):
    if x == 1:
        return 1
    else:
        return x * factorial(x-1)
    

if __name__ == '__main__':
    # unsorted_lst = [5,4,10,25,1,2,3,42]
    # print(selection_sort(unsorted_lst))   
    # countdown(10)
    print(factorial(5))