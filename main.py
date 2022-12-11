from PIL import Image
import numpy as np
import os
import random
import torch
# img_PIL = Image.open('images_evaluation/images_evaluation/Angelic/character01/0965_01.png')
# print(img_PIL)

# img_PIL=np.array(img_PIL)
# print(img_PIL.shape)
# print(os.listdir("images_evaluation/images_evaluation"))
# data_dir='images_evaluation/images_evaluation/'



def data_generator(N,s,q,dir):
    random_nums=[]
    task=[]
    spt_set=[]
    qry_set=[]
    while len(random_nums)<N:#找N个不重复随机数
        a=random.randint(1,659)
        if a not in random_nums:
            random_nums.append(a)
    language=os.listdir(dir)
    character_num=[]
    for l in language:#计算每个字符集的数量
        character_num.append(len(os.listdir(dir+l)))
    for i in range(len(character_num)-1):#将每种语言之前的所有字符含量加到这个语言上，表示到这个语言为止有多少种字符
        character_num[i+1]=character_num[i]+character_num[i+1]

    for i,nums in enumerate(random_nums):
        for index,ch_num in enumerate(character_num):#找随机数对应的字符
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
        for cnt in range(s+q):#字符前s个图片加入到spt中，第s+1到q的图片加入到qry中
            image_dir=character_dir+images_dir_list[cnt]
            image=Image.open(image_dir)
            if cnt < s:
                spt_set.append(image)
            else:
                qry_set.append(image)
    task.append(spt_set)
    task.append(qry_set)
    return task



def data_init(N,s,q):#给数据加label
    task=data_generator(N,s,q,'images_evaluation/images_evaluation/')
    train_data=task[0]
    test_data=task[1]

    new_train_data=[]
    new_test_data=[]

    for i in range(len(train_data)):
        label=[0.,0.,0.,0.,0.]
        label=torch.tensor(label)
        a=[]
        b=[]
        a.append(train_data[i])
        b.append(test_data[i])
        label[i//5]=1.
        a.append(label)#label改成向量形式
        b.append(label)
        new_train_data.append(a)
        new_test_data.append(b)


    train_data=new_train_data
    test_data=new_test_data
    return train_data,test_data