from PIL import Image
import numpy as np
import os
import random
# img_PIL = Image.open('images_evaluation/images_evaluation/Angelic/character01/0965_01.png')
# print(img_PIL)

# img_PIL=np.array(img_PIL)
# print(img_PIL.shape)
# print(os.listdir("images_evaluation/images_evaluation"))
# data_dir='images_evaluation/images_evaluation/'



def data_generator(N,s,q,dir):
    random_nums=[],task=[],spt_set=[],qry_set=[]
    while len(random_nums)<N:
        a=random.randint(1,659)
        if a not in random_nums:
            random_nums.append(a)
    language=os.listdir(dir)
    character_num=[]
    for l in language:
        character_num.append(len(os.listdir(dir+l)))
    for i in range(len(character_num)-1):
        character_num[i+1]=character_num[i]+character_num[i+1]

    for i,nums in enumerate(random_nums):
        for index,ch_num in enumerate(character_num):
            if nums<=ch_num:
                break
        language_dir=dir+language[index]+'/'
        if index>0:
            character_index=nums-character_num[index-1]
        else:
            character_index=nums
        if character_index<10:
            character_dir=language_dir+'character0'+str(character_index)+'/'
        else :
            character_dir=language_dir+'character'+str(character_index)+'/'
        images_dir_list=os.listdir(character_dir)
        for cnt in range(s+q):
            image_dir=character_dir+images_dir_list[cnt]
            image=Image.open(image_dir)
            if cnt < s:
                spt_set.append(image)
            else:
                qry_set.append(image)
    task.append(spt_set)
    task.append(qry_set)
    return task



def data_init(N,s,q):
    task=data_generator(N,s,q,'images_evaluation/images_evaluation/')
    train_data=task[0]
    test_data=task[1]

    new_train_data=[]
    new_test_data=[]

    for i in range(len(train_data)):
        a=[]
        b=[]
        a.append(train_data[i])
        b.append(test_data[i])
        a.append(i//5)
        b.append(i//5)
        new_train_data.append(a)
        new_test_data.append(b)


    train_data=new_train_data
    test_data=new_test_data
    return train_data,test_data