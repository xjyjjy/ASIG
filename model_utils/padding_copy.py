
import torch
import numpy as np
import torch.nn.functional as F

class padding_v1(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.index_lookup = {} # 5 128 ； 6 256 ；7 512； 8 1024
        for level in range(1, 9):
            self.index_lookup[level] = {}
            self.get_level_padding_idx(level)

    def get_level_padding_idx(self, level, pad_l=16):
        pad_u = pad_l*2
        h, w = 4*(2**level), 2**level
        index1_in, index2_in, index3_in, index4_in, index5_in, index6_in = \
            [], [], [], [], [], [] #pink, brown, purple, green, yellow, blue
        index1_out, index2_out, index3_out, index4_out, index5_out, index6_out = \
            [], [], [], [], [], []
        for j in range(pad_u//2):
            for i in range(h//2):
                index1_in.append([4,h//2+pad_u-1 - i,  w+pad_u//2-1 - j ])
        for j in range(pad_u-1, -1, -2):
            for i in range(h//2):
                index1_out.append([0, (i%2)-1 + j, i//2 + pad_u//2 ]) 

        for j in range(pad_u//2):
            for i in range(h//2):
                index2_in.append([4,h+pad_u-1 - i, pad_l+w-1 - j])

        for j in range(pad_u//2):
            for i in range(h//2):
                index2_out.append([0, h//2+pad_u-1 - i, pad_l-1 - j]) #ok

        for j in range(pad_u-1, -1, -2):
            for i in range(h//2):
                index3_in.append([4, h-(i%2) + j , i//2 + pad_u//2 ])

        for j in range(pad_u//2):
            for i in range(h//2):
                index3_out.append([0,h+pad_u-1 - i, pad_l-1 - j]) #ok

        for j in range(pad_u//2):
            for i in range(h//2):
                index4_in.append([1, h+pad_u-1 - i, pad_l+pad_u//2-1 - j ])#ok

        for j in range(pad_u-1, -1, -2):
            for i in range(h//2):
                index4_out.append([0, pad_u + h + (i%2) - 1 + j , i//2 + pad_u//2 ]) #ok

        for j in range(pad_u//2):
            for i in range(h//2):
                index5_in.append([1, h//2+pad_u-1 - i, pad_l*2-1 - j])

        for j in range(pad_u//2):
            for i in range(h//2):
                index5_out.append([0, pad_u+h-1 - i, pad_l*2+w-1 - j]) #ok


        for j in range(pad_u-1, -1, -2):
            for i in range(h//2):
                index6_in.append([1, pad_u-(i%2) + j, i//2 + pad_u//2 ])

        for j in range(pad_u//2):
            for i in range(h//2):
                index6_out.append([0,h//2+pad_u-1 - i, w+pad_l*2-1 - j]) #ok

        index1_in = np.array(index1_in)
        index2_in = np.array(index2_in)
        index3_in = np.array(index3_in)
        index4_in = np.array(index4_in)
        index5_in = np.array(index5_in)
        index6_in = np.array(index6_in)
        index1_out = np.array(index1_out)
        index2_out = np.array(index2_out)
        index3_out = np.array(index3_out)
        index4_out = np.array(index4_out)
        index5_out = np.array(index5_out)
        index6_out = np.array(index6_out)
        index_in = np.concatenate((index1_in, index2_in, index3_in, index4_in, index5_in, index6_in ),axis=0)
        index_out = np.concatenate((index1_out, index2_out, index3_out, index4_out, index5_out, index6_out),axis=0)
        self.index_lookup[level][0] = torch.Tensor(index_out).type(torch.long).to(self.device)
        self.index_lookup[level][1] =  torch.Tensor(index_in).type(torch.long).to(self.device)
    def get_padding(self, input, level):
        input = input.reshape(input.size(0)//5, 5, input.size(1), input.size(2), input.size(3))
        input_pad = F.pad(input, [16,16,32,32])	#padding / [L,R,U,D]
        input_pad_tr = input_pad.permute(0, 1, 3, 4, 2) #change [5,c,h,w] -> [5,h,w,c]
        for i in range(5):
            input_pad_tr[:, (self.index_lookup[level][0][:,0]+i)%5, \
                             self.index_lookup[level][0][:,1], \
                             self.index_lookup[level][0][:,2]] = \
            input_pad_tr[:, (self.index_lookup[level][1][:,0]+i)%5, \
                             self.index_lookup[level][1][:,1], \
                             self.index_lookup[level][1][:,2]]
        result = input_pad_tr.permute(0,1,4,2,3)
        result = result.view(result.size(0)*5,result.size(2),result.size(3),result.size(4))
        # return result[..., 24:-24,8:-8]  # remove padding
        return result[..., 16:-16,::]  # remove padding
    
class padding_v2_0(object): # 保持内容
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.index_lookup = {} # 5 128 ； 6 256 ；7 512； 8 1024
        for level in range(2, 9):
            h, w = 4*(2**level), 2**level
            self.index_lookup[level] = {}
            self.get_level_padding_idx(level, w//2)

    def get_level_padding_idx(self, level, pad_l=16):
        pad_u = pad_l*2
        h, w = 4*(2**level), 2**level
        index1_in, index2_in, index3_in, index4_in, index5_in, index6_in = \
            [], [], [], [], [], [] #pink, brown, purple, green, yellow, blue
        index1_out, index2_out, index3_out, index4_out, index5_out, index6_out = \
            [], [], [], [], [], []
        for j in range(pad_u//2):
            for i in range(h//2 - (2*j+1)):
                index1_in.append([4,(pad_u+h//2-1-(2*j+1)) - i,  w+pad_l-1 - j ])
            column_index = w+pad_l-1
            for i in range(2*j+1):
                index1_in.append([3, pad_u+2*j - i , column_index])
                if i%2 == 1:
                    column_index -= 1
                    
        for j in range(pad_u-1, -1, -2):
            for i in range(h//2):
                index1_out.append([0, -(i%2) + j, i//2 + pad_u//2 ])  #ok_0
        # print("index_in", len(index1_in), "index_out", len(index1_out))

        for j in range(pad_u//2):
            for i in range(h//2):
                index2_in.append([4,h+pad_u-1 - i -2*j-1, pad_l+w-1 - j])

        for j in range(pad_u//2):
            for i in range(h//2):
                index2_out.append([0, h//2+pad_u-1 - i, pad_l-1 - j]) # ok_0
                


        for j in range(pad_u-1, -1, -2):
            column_index = pad_l 
            column_count = (pad_u-1-j)//2
            for i in range(2*column_count+1):
                index3_in.append([3, pad_u+h-1 - 2*column_count  + i , column_index])
                if i%2 == 1:
                    column_index += 1
            for i in range(h//2 - (2*column_count+1)):
                index3_in.append([4, h-(i%2) + j , i//2 + pad_l ])

        for j in range(pad_u//2):
            for i in range(h//2):
                index3_out.append([0,h+pad_u-1 - i, pad_l-1 - j]) #ok_0


        for j in range(pad_u//2):
            column_index = pad_l + j
            for i in range(2*j+1):
                index4_in.append([2, pad_u+h-1 - i , column_index])
                if i%2 == 0:
                    column_index -= 1

            for i in range(h//2 - (2*j+1)):
                index4_in.append([1,(pad_u+h-1) - i,  pad_l + j ])
            
                    
        for j in range(0, pad_u, 2):
            for i in range(h//2):
                index4_out.append([0, pad_u+h + 1-(i%2) + j, i//2 + pad_l ])  #ok_0

        for j in range(pad_u//2):
            for i in range(h//2):
                index5_in.append([1, pad_u+1 + i + 2*j, pad_l + j])

        for j in range(pad_u//2):
            for i in range(h//2):
                index5_out.append([0, pad_u+h//2+ i, pad_l+w + j]) #ok
                
        for j in range(pad_u-1, -1, -2):
            column_index = w + pad_l - 1
            column_count = (pad_u-1-j)//2
            for i in range(2*column_count+1):
                index6_in.append([2, pad_u + 2*column_count - i, column_index])
                if i%2 == 1:
                    column_index -= 1
            
            column_index = pad_l + w - 1
            for i in range(h//2 - (2*column_count+1)):
                index6_in.append([1, pad_u + 2*column_count + (i%2), column_index])
                if i%2 == 1:
                    column_index -= 1

        for j in range(pad_u//2):
            for i in range(h//2):
                index6_out.append([0, pad_u + i, w+pad_l + j]) #ok_0

        index1_in = np.array(index1_in)
        index2_in = np.array(index2_in)
        index3_in = np.array(index3_in)
        index4_in = np.array(index4_in)
        index5_in = np.array(index5_in)
        index6_in = np.array(index6_in)

        index1_out = np.array(index1_out)
        index2_out = np.array(index2_out)
        index3_out = np.array(index3_out)
        index4_out = np.array(index4_out)
        index5_out = np.array(index5_out)
        index6_out = np.array(index6_out)
        index_in = np.concatenate((index1_in, index2_in, index3_in, index4_in, index5_in, index6_in ),axis=0)
        index_out = np.concatenate((index1_out, index2_out, index3_out, index4_out, index5_out, index6_out),axis=0)
        self.index_lookup[level][0] = torch.Tensor(index_out).type(torch.long).to(self.device)
        self.index_lookup[level][1] =  torch.Tensor(index_in).type(torch.long).to(self.device)
    def get_padding(self, input, level):
        h, w = 4*(2**level), 2**level
        pad_size = w//4
        input = input.reshape(input.size(0)//5, 5, input.size(1), input.size(2), input.size(3))
        input_pad = F.pad(input, [pad_size,pad_size,pad_size*2,pad_size*2])	#padding / [L,R,U,D]
        input_pad_tr = input_pad.permute(0, 1, 3, 4, 2) #change [5,c,h,w] -> [5,h,w,c]
        for i in range(5):
            input_pad_tr[:, (self.index_lookup[level][0][:,0]+i)%5, \
                             self.index_lookup[level][0][:,1], \
                             self.index_lookup[level][0][:,2]] = \
            input_pad_tr[:, (self.index_lookup[level][1][:,0]+i)%5, \
                             self.index_lookup[level][1][:,1], \
                             self.index_lookup[level][1][:,2]]
        result = input_pad_tr.permute(0,1,4,2,3)
        result = result.view(result.size(0)*5,result.size(2),result.size(3),result.size(4))
        return result  # remove padding
    def return_noise(self, input, level):
        input = input.reshape(input.size(0)//5, 5, input.size(1), input.size(2), input.size(3))
        input_pad_tr = input.permute(0, 1, 3, 4, 2) #change [5,c,h,w] -> [5,h,w,c]
        input_pad_tr_clone = input_pad_tr.clone()
        alpha = 0.5
        for i in range(5):
            input_pad_tr[:, (self.index_lookup[level][1][:,0]+i)%5, \
                             self.index_lookup[level][1][:,1], \
                             self.index_lookup[level][1][:,2]] = \
            alpha*input_pad_tr_clone[:, (self.index_lookup[level][0][:,0]+i)%5, \
                             self.index_lookup[level][0][:,1], \
                             self.index_lookup[level][0][:,2]] + \
            (1-alpha)*input_pad_tr_clone[:, (self.index_lookup[level][1][:,0]+i)%5, \
                             self.index_lookup[level][1][:,1], \
                             self.index_lookup[level][1][:,2]]


            # input_pad_tr[:, (self.index_lookup[level][0][:,0]+i)%5, \
            #                  self.index_lookup[level][0][:,1], \
            #                  self.index_lookup[level][0][:,2]] = \
            # input_pad_tr[:, (self.index_lookup[level][1][:,0]+i)%5, \
            #                  self.index_lookup[level][1][:,1], \
            #                  self.index_lookup[level][1][:,2]]
        result = input_pad_tr.permute(0,1,4,2,3)
        result = result.view(result.size(0)*5,result.size(2),result.size(3),result.size(4))
        return result  # remove padding
    
class padding_v2_1(object):  ## 平移
    def __init__(self, overlap_scale = 2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.index_lookup = {} # 5 128 ； 6 256 ；7 512； 8 1024
        self.overlap_scale = overlap_scale
        for level in [2,3,4,5,8]:
            h, w = 4*(2**level), 2**level
            self.index_lookup[level] = {}
            self.get_level_padding_idx(level, w//overlap_scale)

    def get_level_padding_idx(self, level, pad_l=16):
        pad_u = pad_l*2
        h, w = 4*(2**level), 2**level
        index1_in, index2_in, index3_in, index4_in, index5_in, index6_in = \
            [], [], [], [], [], [] #pink, brown, purple, green, yellow, blue
        index1_out, index2_out, index3_out, index4_out, index5_out, index6_out = \
            [], [], [], [], [], []
        for j in range(pad_u//2):
            for i in range(h//2 - (2*j+1)):
                index1_in.append([4,(pad_u+h//2-1-(2*j+1)) - i,  w+pad_l-1 - j ])
            column_index = w+pad_l-1
            for i in range(2*j+1):
                index1_in.append([3, pad_u+2*j - i , column_index])
                if i%2 == 1:
                    column_index -= 1
                    
        for j in range(pad_u-1, -1, -2):
            for i in range(h//2):
                index1_out.append([0, -(i%2) + j, i//2 + pad_u//2 ])  #ok_0
        # print("index_in", len(index1_in), "index_out", len(index1_out))

        # for j in range(pad_u//2):
        #     for i in range(h//2):
        #         index2_in.append([4,h+pad_u-1 - i -2*j-1, pad_l+w-1 - j])

        # for j in range(pad_u//2):
        #     for i in range(h//2):
        #         index2_out.append([0, h//2+pad_u-1 - i, pad_l-1 - j]) # ok_0
                
        for j in range(pad_u//2):
            for i in range(h//2):
                index2_in.append([4,h+pad_u-1 - i, pad_l+w-1 - j])

        for j in range(pad_u//2):
            for i in range(h//2):
                index2_out.append([0, h//2+pad_u-1 - i, pad_l-1 - j]) #ok


        for j in range(pad_u-1, -1, -2):
            column_index = pad_l 
            column_count = (pad_u-1-j)//2
            for i in range(2*column_count+1):
                index3_in.append([3, pad_u+h-1 - 2*column_count  + i , column_index])
                if i%2 == 1:
                    column_index += 1
            for i in range(h//2 - (2*column_count+1)):
                index3_in.append([4, h-(i%2) + j , i//2 + pad_l ])

        for j in range(pad_u//2):
            for i in range(h//2):
                index3_out.append([0,h+pad_u-1 - i, pad_l-1 - j]) #ok_0


        for j in range(pad_u//2):
            column_index = pad_l + j
            for i in range(2*j+1):
                index4_in.append([2, pad_u+h-1 - i , column_index])
                if i%2 == 0:
                    column_index -= 1

            for i in range(h//2 - (2*j+1)):
                index4_in.append([1,(pad_u+h-1) - i,  pad_l + j ])
            
                    
        for j in range(0, pad_u, 2):
            for i in range(h//2):
                index4_out.append([0, pad_u+h + 1-(i%2) + j, i//2 + pad_l ])  #ok_0

        # for j in range(pad_u//2):
        #     for i in range(h//2):
        #         index5_in.append([1, pad_u+1 + i + 2*j, pad_l + j])

        # for j in range(pad_u//2):
        #     for i in range(h//2):
        #         index5_out.append([0, pad_u+h//2+ i, pad_l+w + j]) #ok
                
        for j in range(pad_u//2):
            for i in range(h//2):
                index5_in.append([1, h//2+pad_u-1 - i, pad_l*2-1 - j])

        for j in range(pad_u//2):
            for i in range(h//2):
                index5_out.append([0, pad_u+h-1 - i, pad_l*2+w-1 - j]) #ok


        for j in range(pad_u-1, -1, -2):
            column_index = w + pad_l - 1
            column_count = (pad_u-1-j)//2
            for i in range(2*column_count+1):
                index6_in.append([2, pad_u + 2*column_count - i, column_index])
                if i%2 == 1:
                    column_index -= 1
            
            column_index = pad_l + w - 1
            for i in range(h//2 - (2*column_count+1)):
                index6_in.append([1, pad_u + 2*column_count + (i%2), column_index])
                if i%2 == 1:
                    column_index -= 1

        for j in range(pad_u//2):
            for i in range(h//2):
                index6_out.append([0, pad_u + i, w+pad_l + j]) #ok_0

        index1_in = np.array(index1_in)
        index2_in = np.array(index2_in)
        index3_in = np.array(index3_in)
        index4_in = np.array(index4_in)
        index5_in = np.array(index5_in)
        index6_in = np.array(index6_in)

        index1_out = np.array(index1_out)
        index2_out = np.array(index2_out)
        index3_out = np.array(index3_out)
        index4_out = np.array(index4_out)
        index5_out = np.array(index5_out)
        index6_out = np.array(index6_out)
        index_in = np.concatenate((index1_in, index2_in, index3_in, index4_in, index5_in, index6_in ),axis=0)
        index_out = np.concatenate((index1_out, index2_out, index3_out, index4_out, index5_out, index6_out),axis=0)
        self.index_lookup[level][0] = torch.Tensor(index_out).type(torch.long).to(self.device)
        self.index_lookup[level][1] =  torch.Tensor(index_in).type(torch.long).to(self.device)
    def get_padding(self, input, level):
        h, w = 4*(2**level), 2**level
        pad_size = w//self.overlap_scale
        input = input.reshape(input.size(0)//5, 5, input.size(1), input.size(2), input.size(3))
        input_pad = F.pad(input, [pad_size,pad_size,pad_size*2,pad_size*2])	#padding / [L,R,U,D]
        input_pad_tr = input_pad.permute(0, 1, 3, 4, 2) #change [5,c,h,w] -> [5,h,w,c]
        for i in range(5):
            input_pad_tr[:, (self.index_lookup[level][0][:,0]+i)%5, \
                             self.index_lookup[level][0][:,1], \
                             self.index_lookup[level][0][:,2]] = \
            input_pad_tr[:, (self.index_lookup[level][1][:,0]+i)%5, \
                             self.index_lookup[level][1][:,1], \
                             self.index_lookup[level][1][:,2]]
        result = input_pad_tr.permute(0,1,4,2,3)
        result = result.view(result.size(0)*5,result.size(2),result.size(3),result.size(4))
        # return result
        # print("result shape", result.shape)
        # return result
        if self.overlap_scale == 2:
            return result[...,pad_size:-pad_size, ::]  # remove padding
        else:
            return result
    def return_noise(self, input, level):
        input = input.reshape(input.size(0)//5, 5, input.size(1), input.size(2), input.size(3))
        input_pad_tr = input.permute(0, 1, 3, 4, 2) #change [5,c,h,w] -> [5,h,w,c]
        input_pad_tr_clone = input_pad_tr.clone()
        alpha = 0.5
        for i in range(5):
            input_pad_tr[:, (self.index_lookup[level][1][:,0]+i)%5, \
                             self.index_lookup[level][1][:,1], \
                             self.index_lookup[level][1][:,2]] = \
            alpha*input_pad_tr_clone[:, (self.index_lookup[level][0][:,0]+i)%5, \
                             self.index_lookup[level][0][:,1], \
                             self.index_lookup[level][0][:,2]] + \
            (1-alpha)*input_pad_tr_clone[:, (self.index_lookup[level][1][:,0]+i)%5, \
                             self.index_lookup[level][1][:,1], \
                             self.index_lookup[level][1][:,2]]


            # input_pad_tr[:, (self.index_lookup[level][0][:,0]+i)%5, \
            #                  self.index_lookup[level][0][:,1], \
            #                  self.index_lookup[level][0][:,2]] = \
            # input_pad_tr[:, (self.index_lookup[level][1][:,0]+i)%5, \
            #                  self.index_lookup[level][1][:,1], \
            #                  self.index_lookup[level][1][:,2]]
        result = input_pad_tr.permute(0,1,4,2,3)
        result = result.view(result.size(0)*5,result.size(2),result.size(3),result.size(4))
        return result  # remove padding
    
    
class padding(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.index_lookup = {}
        for level in range(1, 8):
            self.index_lookup[level] = {}
            self.get_level_padding_idx(level)

    def get_level_padding_idx(self, level):
        h, w = 4*(2**level), 2**level
        index1_in, index2_in, index3_in, index4_in, index5_in, index6_in = \
            [], [], [], [], [], [] #pink, brown, purple, green, yellow, blue
        index1_out, index2_out, index3_out, index4_out, index5_out, index6_out = \
            [], [], [], [], [], []

        for i in range(h//2):
            index1_in.append([4,h//2-1-i +2,w-1 + 1])
        index1_in.append([3,0+2,w-1+1])

        for i in range(h//2+2):
            if i == 0:
                continue
            else :
                index1_out.append([0, (i+1)%2, i//2 ])

        for i in range(h//2 + 1):
            index2_in.append([4,h -1 - i+2,w-1+1])

        for i in range(h//2 + 1):
            index2_out.append([0, (h//2 + 2) - 1 - i, 0])

        index3_in.append([3,h-1 +2,0+1])
        for i in range(w-1):
            index3_in.append([4, h-1+2, i+1 ])
            index3_in.append([4, h-2+2, i+1 ])
        index3_in.append([4, h-1+2, w-1+1])

        for i in range(h//2):
            index3_out.append([0, (h + 2) -1 - i, 0])

        for i in range(h//2):
            index4_in.append([1,h+3 - (h//2-1-i +2), w+ 1 -(w-1 + 1)])
        index4_in.append([2,h+3 -(0+2),w+1 - (w-1+1)])

        for i in range(h//2+2):
            if i == 0:
                continue
            else :
                index4_out.append([0, h+3 - ((i+1)%2), w+1 - (i//2) ])


        for i in range(h//2 + 1):
            index5_in.append([1,h+3 - (h -1 - i+2),w+1 - (w-1+1)])

        for i in range(h//2 + 1):
            index5_out.append([0,h+3 - ((h//2 + 2) - 1 - i), w+1 - 0])


        index6_in.append([2,h+3 - (h-1+2),w+1 - (0+1)])
        for i in range(w-1):
            index6_in.append([1, h+3 - (h-1+2), w+1 - (i+1) ])
            index6_in.append([1, h+3 - (h-2+2), w+1 - (i+1) ])
        index6_in.append([1, h+3 - (h-1+2), w+1 - (w-1+1)])

        for i in range(h//2):
            index6_out.append([0, h+3 - ((h + 2) -1 - i), w+1 - 0])

        index1_in = np.array(index1_in)
        index2_in = np.array(index2_in)
        index3_in = np.array(index3_in)
        index4_in = np.array(index4_in)
        index5_in = np.array(index5_in)
        index6_in = np.array(index6_in)

        index1_out = np.array(index1_out)
        index2_out = np.array(index2_out)
        index3_out = np.array(index3_out)
        index4_out = np.array(index4_out)
        index5_out = np.array(index5_out)
        index6_out = np.array(index6_out)

        index_in = np.concatenate((index1_in, index2_in, index3_in, index4_in, index5_in, index6_in),axis=0)
        index_out = np.concatenate((index1_out, index2_out, index3_out, index4_out, index5_out, index6_out),axis=0)

        self.index_lookup[level][0] = torch.Tensor(index_out).type(torch.long).to(self.device)
        self.index_lookup[level][1] =  torch.Tensor(index_in).type(torch.long).to(self.device)


    def get_padding(self, input, level, flag): # common_conv 0, sphere_conv 1
        input = input.reshape(input.size(0)//5, 5, input.size(1), input.size(2), input.size(3))
        input_pad = F.pad(input, [1,1,2,2])	#padding / [L,R,U,D]
        input_pad_tr = input_pad.permute(0, 1, 3, 4, 2) #change [5,c,h,w] -> [5,h,w,c]
        for i in range(5):
            input_pad_tr[:, (self.index_lookup[level][0][:,0]+i)%5, \
                             self.index_lookup[level][0][:,1], \
                             self.index_lookup[level][0][:,2]] = \
            input_pad_tr[:, (self.index_lookup[level][1][:,0]+i)%5, \
                             self.index_lookup[level][1][:,1], \
                             self.index_lookup[level][1][:,2]]
        result = input_pad_tr.permute(0,1,4,2,3)
        result = result.view(result.size(0)*5,result.size(2),result.size(3),result.size(4))
        if flag == 0:
            return result[:,:,1:-1,:]
        else:
            return result  
        
    def return_padding(self, input, level, flag): # common_conv 0, sphere_conv 1
        input = input.reshape(input.size(0)//5, 5, input.size(1), input.size(2), input.size(3))
        # input_pad = F.pad(input, [1,1,2,2])	#padding / [L,R,U,D]
        input_pad_tr = input.permute(0, 1, 3, 4, 2) #change [5,c,h,w] -> [5,h,w,c]
        input_pad_tr_save = input_pad_tr.clone()
        for i in range(5):
            input_pad_tr[:, (self.index_lookup[level][1][:,0]+i)%5, \
                             self.index_lookup[level][1][:,1], \
                             self.index_lookup[level][1][:,2]] = \
            0.5 * input_pad_tr_save[:, (self.index_lookup[level][0][:,0]+i)%5, \
                             self.index_lookup[level][0][:,1], \
                             self.index_lookup[level][0][:,2]] + \
            0.5 * input_pad_tr_save[:, (self.index_lookup[level][1][:,0]+i)%5, \
                             self.index_lookup[level][1][:,1], \
                             self.index_lookup[level][1][:,2]]
            
        result = input_pad_tr.permute(0,1,4,2,3)
        result = result.view(result.size(0)*5,result.size(2),result.size(3),result.size(4))

        del input_pad_tr_save

        return result[:,:,2:-2, 1:-1]

class padding_old(object):
    def __init__(self, level):
        self.level = level
        h, w = 4*(2**self.level), 2**self.level

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        index1_in = [] #pink
        index2_in = [] #brown
        index3_in = [] #purple
        index4_in = [] #green
        index5_in = [] #yellow
        index6_in = [] #blue

        index1_out = []
        index2_out = []
        index3_out = []
        index4_out = []
        index5_out = []
        index6_out = []

        for i in range(h//2):
            index1_in.append([4,h//2-1-i +2,w-1 + 1])
        index1_in.append([3,0+2,w-1+1])

        for i in range(h//2+2):
            if i == 0:
                    continue
            else :
                    index1_out.append([0, (i+1)%2, i//2 ])


        for i in range(h//2 + 1):
            index2_in.append([4,h -1 - i+2,w-1+1])

        for i in range(h//2 + 1):
            index2_out.append([0, (h//2 + 2) - 1 - i, 0])


        index3_in.append([3,h-1 +2,0+1])
        for i in range(w-1):
            index3_in.append([4, h-1+2, i+1 ])
            index3_in.append([4, h-2+2, i+1 ])
        index3_in.append([4, h-1+2, w-1+1])

        for i in range(h//2):
            index3_out.append([0, (h + 2) -1 - i, 0])
        


        for i in range(h//2):
            index4_in.append([1,h+3 - (h//2-1-i +2), w+ 1 -(w-1 + 1)])
        index4_in.append([2,h+3 -(0+2),w+1 - (w-1+1)])

        for i in range(h//2+2):
            if i == 0:
                continue
            else :
                index4_out.append([0, h+3 - ((i+1)%2), w+1 - (i//2) ])


        for i in range(h//2 + 1):
            index5_in.append([1,h+3 - (h -1 - i+2),w+1 - (w-1+1)])

        for i in range(h//2 + 1):
            index5_out.append([0,h+3 - ((h//2 + 2) - 1 - i), w+1 - 0])


        index6_in.append([2,h+3 - (h-1+2),w+1 - (0+1)])
        for i in range(w-1):
            index6_in.append([1, h+3 - (h-1+2), w+1 - (i+1) ])
            index6_in.append([1, h+3 - (h-2+2), w+1 - (i+1) ])
        index6_in.append([1, h+3 - (h-1+2), w+1 - (w-1+1)])

        for i in range(h//2):
            index6_out.append([0, h+3 - ((h + 2) -1 - i), w+1 - 0])

        index1_in = np.array(index1_in)
        index2_in = np.array(index2_in)
        index3_in = np.array(index3_in)
        index4_in = np.array(index4_in)
        index5_in = np.array(index5_in)
        index6_in = np.array(index6_in)

        index1_out = np.array(index1_out)
        index2_out = np.array(index2_out)
        index3_out = np.array(index3_out)
        index4_out = np.array(index4_out)
        index5_out = np.array(index5_out)
        index6_out = np.array(index6_out)

        self.index_in = np.concatenate((index1_in,index2_in,index3_in,index4_in,index5_in,index6_in),axis=0)
        self.index_out = np.concatenate((index1_out,index2_out,index3_out,index4_out,index5_out,index6_out),axis=0)

        self.index_in = torch.Tensor(self.index_in).type(torch.long).to(self.device)
        self.index_out = torch.Tensor(self.index_out).type(torch.long).to(self.device)


    def get_padding(self, input):
        input_pad = F.pad(input, [1,1,2,2])	#padding / [L,R,U,D]
        input_pad_tr = input_pad.permute(0, 1, 3, 4, 2) #change [5,c,h,w] -> [5,h,w,c]
        for i in range(5):
            input_pad_tr[:, (self.index_out[:,0]+i)%5, self.index_out[:,1], self.index_out[:,2]] = \
            input_pad_tr[:, (self.index_in[:,0]+i)%5, self.index_in[:,1], self.index_in[:,2]]
        input_pad_result = input_pad_tr.permute(0,1,4,2,3)
        return input_pad_result


def check_time():
    import time
    level = 6
    h, w = 4*(2**level), 2**level
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pad_level = padding(level)
    input = torch.randn(100*5*h*w).view(100,5,1,h,w).to(device)  #sub_img_num, feature_num, height, width
    start = time.time() 
    result = pad_level.get_padding(input)
    # result = pad_level.get_padding2(input)
    resume_time_new = time.time() - start
    print("time(new) :", resume_time_new)
    # print(pad_level2.get_padding(input))
    x0 = torch.zeros(input.size(0),input.size(1),input.size(2),input.size(3)+4,input.size(4)+2).cuda()
    start = time.time() 
    for idx, x in enumerate(input):
        x0[idx] = pad_level.get_padding_old(x)
    resume_time_old = time.time() - start
    print("time(old) :", resume_time_old)
    print("speed ratio: ", resume_time_old/resume_time_new)
    if ((~(x0==result)).sum())==0:
        print('same result')
    else:
        print('different result')


class padding_v3_0(object):
    def __init__(self, list_level):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.index_lookup = {} # 5 128 ； 6 256 ；7 512； 8 1024
        for level in list_level:
            self.index_lookup[level] = {}
            self.get_level_padding_idx(level)

    def get_right_bottom_out(self, h, w, flag, patch=0):
        if flag == 1:
            index_out = []
            for i in range(h//2 - 1):
                for j in range(i + 1):
                    raw = h//2 + 1 + i
                    col = w - 1 - i + j 
                    index_out.append([patch, raw, col])
            return np.array(index_out)
        elif flag == 0:
            index_out = []
            for i in range(h//2 - 1):
                for j in range(i + 1):
                    raw = h//2 + 1 + i
                    col = w - 1 - i + j 
                    index_out.append([patch, raw, col])
            return np.array(index_out)
    
    def get_right_bottom_in_0(self, h, w, flag):
        if flag == 1:
            index_in = []
            for i in range(h//2 - 1):
                for j in range(i + 1):
                    raw = i
                    col = w//2 - i + j 
                    index_in.append([1, raw, col])
            return np.array(index_in)
        elif flag == 0:
            index_in = []
            for i in range(h//2 - 1):
                for j in range(i + 1):
                    raw = i
                    col = w//2 - 1 - i + j 
                    index_in.append([1, raw, col])
            return np.array(index_in)
    
    def get_left_top_out(self, h, w, flag, patch=0):
        if flag == 1:
            index_out = []
            for i in range(h//2 - 1):
                for j in range(w//2 - i):
                    raw = i
                    col = j
                    index_out.append([patch, raw, col])
            return np.array(index_out)

    def get_left_top_in_1(self, h, w, flag):
        if flag == 1:
            index_in = []
            for i in range(h//2 - 1):
                for j in range(w//2 - i):
                    raw = h//2 + 1 + i
                    col = w//2 + j
                    index_in.append([0, raw, col])
            return np.array(index_in)
        elif flag == 0:
            index_in = []
            for i in range(h//2 - 1):
                for j in range(w//2 - 1 - i):
                    raw = h//2 + 1 + i
                    col = w//2 + j
                    index_in.append([0, raw, col])
            return np.array(index_in)

        
    def get_right_top_out(self, h, w, flag, patch=0):
        if flag == 1:
            index_out = []
            for i in range(h//2 - 1):
                for j in range(w//2 - i):
                    raw = i
                    col = w//2 + i + 1 + j
                    index_out.append([patch, raw, col])
            return np.array(index_out)
        elif flag == 0:
            index_out = []
            for i in range(h//2 - 1):
                for j in range(w//2 - 1 - i):
                    raw = i
                    col = w//2 + i + j + 1
                    index_out.append([patch, raw, col])
            return np.array(index_out)
    
    def get_right_top_in_1(self, h, w, flag):
        if flag == 1:
            index_in = []
            for i in range(h//2 - 1):
                for j in range(w//2 - i):
                    raw = h//2 + i
                    col = i + j
                    index_in.append([2, raw, col])
            return np.array(index_in)
        elif flag == 0:
            index_in = []
            for i in range(h//2 - 1):
                for j in range(w//2 - 1 - i):
                    raw = h//2 + i + 1
                    col = i + j + 1
                    index_in.append([2, raw, col])
            return np.array(index_in)
    
    def get_left_bottom_out(self, h, w, flag, patch=0):
        if flag == 1:
            index_out = []
            for i in range(h//2 - 1):
                for j in range(i + 1):
                    raw = h//2 + 1 + i
                    col = j
                    index_out.append([patch, raw, col])
            return np.array(index_out)
        elif flag == 0:
            index_out = []
            for i in range(h//2 - 1):
                for j in range(i + 1):
                    raw = h//2 + 1 + i
                    col = j
                    index_out.append([patch, raw, col])
            return np.array(index_out)
        
    def get_left_bottom_in_0(self, h, w, flag):
        if flag == 1:
            index_in = []     
            for i in range(h//2-1):
                for j in range(i + 1):
                    raw = i
                    col = w//2 + j
                    index_in.append([9, raw, col])
            return np.array(index_in)
        elif flag == 0:
            index_in = []     
            for i in range(h//2-1):
                for j in range(i + 1):
                    raw = i
                    col = w//2 + j
                    index_in.append([9, raw, col])
            return np.array(index_in)
            

    def get_left_top_in_0(self, h, w, flag):
        if flag == 1:
            index_in = []
            for j in range(w//2 + 1):
                for i in range(h//2 - 1 - j):
                    raw = h//2 - 2 - i
                    col = w//2 + j
                    index_in.append([8, raw, col])
            return np.array(index_in)
        elif flag == 0:
            index_in = []
            for j in range(w//2 - 1):
                for i in range(h//2 - 1 - j):
                    raw = h//2 - 2 - i
                    col = w//2 + j
                    index_in.append([8, raw, col])
            return np.array(index_in)
        
    def get_left_top_out(self, h, w, flag, patch=0):
        if flag == 1:
            index_out = []
            for i in range(h//2 - 1):
                for j in range(w//2 - i):
                    raw = i
                    col = j
                    index_out.append([patch, raw, col])
            return np.array(index_out)
        
        elif flag == 0:
            index_out = []
            for i in range(h//2 - 1):
                for j in range(w//2 - 1 - i):
                    raw = i
                    col = j
                    index_out.append([patch, raw, col])
            return np.array(index_out)
    
    def get_right_top_in_0(self, h, w, flag):
        if flag == 1:
            index_in = []
            for j in range(w//2):
                for i in range(h//2 - 1 - j):
                    raw = i + j
                    col = w//2 -j
                    index_in.append([2, raw, col])
            return np.array(index_in)
        elif flag == 0:
            index_in = []
            for j in range(w//2 - 1):
                for i in range(h//2 - 1 - j):
                    raw = i + j 
                    col = w//2 - j - 1
                    index_in.append([2, raw, col])
            return np.array(index_in)
        
    def get_left_bottom_in_1(self, h, w, flag):
        if flag == 1:
            index_in = []
            for j in range(w//2):
                for i in range(j+1):
                    raw = h//2 + i
                    col = w - 1 - j
                    index_in.append([9, raw, col])
            return np.array(index_in)
        elif flag == 0:
            index_in = []
            for j in range(w//2 - 1):
                for i in range(j+1):
                    raw = h//2 + i + 1
                    col = w - 2 - j
                    index_in.append([9, raw, col])
            return np.array(index_in)

    def get_right_bottom_in_1(self, h, w, flag):
        if flag == 1:
            index_in = []
            for j in range(w//2):
                for i in range(j+1):
                    raw = h//2 + j - i
                    col = j + 1
                    index_in.append([3, raw, col])
            return np.array(index_in)
        elif flag == 0:
            index_in = []
            for j in range(w//2 - 1):
                for i in range(j+1):
                    raw = h//2 + j - i + 1
                    col = j + 1
                    index_in.append([3, raw, col])
            return np.array(index_in)
                    
    def get_level_padding_idx(self, level, flag = 0):

        h = 2**(level+1)
        w = 2**(level + 1)

        index0_out = self.get_right_bottom_out(h, w, 0)
        index0_in = self.get_right_bottom_in_0(h, w, 0)

        index1_out = self.get_left_bottom_out(h, w, 0)
        index1_in = self.get_left_bottom_in_0(h, w, 0)

        index2_out = self.get_left_top_out(h, w, 0, patch=1)
        index2_in = self.get_left_top_in_1(h,w, 0)

        index3_out = self.get_right_top_out(h, w, 0, patch=1)
        index3_in = self.get_right_top_in_1(h, w, 0)

        index4_out = self.get_left_top_out(h, w, 0)
        index4_in = self.get_left_top_in_0(h, w, 0)

        index5_out = self.get_right_top_out(h, w, 0)
        index5_in = self.get_right_top_in_0(h, w, 0)

        index6_out = self.get_left_bottom_out(h, w, 0, patch=1)
        index6_in = self.get_left_bottom_in_1(h, w, 0)

        index7_out = self.get_right_bottom_out(h, w, 0, patch=1)
        index7_in = self.get_right_bottom_in_1(h, w, 0) 
            

        index_in = np.concatenate((index0_in, index1_in, index2_in, index3_in, index4_in, index5_in, index6_in, index7_in),axis=0)
        index_out = np.concatenate((index0_out, index1_out, index2_out, index3_out, index4_out, index5_out, index6_out, index7_out),axis=0)

        self.index_lookup[level][0] = torch.Tensor(index_out).type(torch.long).to(self.device)
        self.index_lookup[level][1] =  torch.Tensor(index_in).type(torch.long).to(self.device)

    def get_padding(self, input, level):
        # input_pad = F.pad(input, [1,1,0,0])	#padding / [L,R,U,D]
        bs, c, h, w = input.shape
        input = input.view(bs//10, 10, c, h, w)  # reshape to [bs, 5, c, h, w]
        input_tr = input.permute(0, 1, 3, 4, 2) #change [5,c,h,w] -> [5,h,w,c]
        for i in range(0, 10, 2):
            input_tr[:, (self.index_lookup[level][0][:,0]+i)%10, self.index_lookup[level][0][:,1],  self.index_lookup[level][0][:,2]] = \
            input_tr[:, (self.index_lookup[level][1][:,0]+i)%10, self.index_lookup[level][1][:,1],  self.index_lookup[level][1][:,2]]
        result = input_tr.permute(0,1,4,2,3)
        result = result.view(result.size(0)*10,result.size(2),result.size(3),result.size(4))
        # return result[..., 24:-24,8:-8]  # remove padding
        return result  # remove padding
    
    def return_padding(self, input, level):
        bs, c, h, w = input.shape
        input = input.view(bs//10, 10, c, h, w)  # reshape to [bs, 5, c, h, w]
        input_tr = input.permute(0, 1, 3, 4, 2) #change [5,c,h,w] -> [5,h,w,c]
        for i in range(0, 10, 2):
            # input_tr[:, (self.index_lookup[level][0][:,0]+i)%10, self.index_lookup[level][0][:,1],  self.index_lookup[level][0][:,2]] = \
            # input_tr[:, (self.index_lookup[level][1][:,0]+i)%10, self.index_lookup[level][1][:,1],  self.index_lookup[level][1][:,2]]
            input_tr[:, (self.index_lookup[level][1][:,0]+i)%10, self.index_lookup[level][1][:,1],  self.index_lookup[level][1][:,2]] = \
            input_tr[:, (self.index_lookup[level][0][:,0]+i)%10, self.index_lookup[level][0][:,1],  self.index_lookup[level][0][:,2]] 
            # 0.5 * input_tr[:, (self.index_lookup[level][1][:,0]+i)%10, self.index_lookup[level][1][:,1],  self.index_lookup[level][1][:,2]]
        result = input_tr.permute(0,1,4,2,3)
        result = result.view(result.size(0)*10,result.size(2),result.size(3),result.size(4))
        return result  # remove padding

class padding_v3_1(object):
    def __init__(self, list_level):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.index_lookup = {} # 5 128 ； 6 256 ；7 512； 8 1024
        for level in list_level:
            self.index_lookup[level] = {}
            self.get_level_padding_idx(level)

    def get_right_bottom_out(self, h, w, flag, patch=0):
        if flag == 1:
            index_out = []
            for i in range(h//2 - 1):
                for j in range(i + 1):
                    raw = h//2 + 1 + i
                    col = w - 1 - i + j 
                    index_out.append([patch, raw, col])
            return np.array(index_out)
        elif flag == 0:
            index_out = []
            for i in range(h//2 - 1):
                for j in range(i + 1):
                    raw = h//2 + 1 + i
                    col = w - 1 - i + j 
                    index_out.append([patch, raw, col])
            return np.array(index_out)
    
    def get_right_bottom_in_0(self, h, w, flag):
        if flag == 1:
            index_in = []
            for i in range(h//2 - 1):
                for j in range(i + 1):
                    raw = i
                    col = w//2 - i + j 
                    index_in.append([1, raw, col])
            return np.array(index_in)
        elif flag == 0:
            index_in = []
            for i in range(h//2 - 1):
                for j in range(i + 1):
                    raw = i
                    col = w//2 - 1 - i + j 
                    index_in.append([1, raw, col])
            return np.array(index_in)
    
    def get_left_top_out(self, h, w, flag, patch=0):
        if flag == 1:
            index_out = []
            for i in range(h//2 - 1):
                for j in range(w//2 - i):
                    raw = i
                    col = j
                    index_out.append([patch, raw, col])
            return np.array(index_out)

    def get_left_top_in_1(self, h, w, flag):
        if flag == 1:
            index_in = []
            for i in range(h//2 - 1):
                for j in range(w//2 - i):
                    raw = h//2 + 1 + i
                    col = w//2 + j
                    index_in.append([0, raw, col])
            return np.array(index_in)
        elif flag == 0:
            index_in = []
            for i in range(h//2 - 1):
                for j in range(w//2 - 1 - i):
                    raw = h//2 + 1 + i
                    col = w//2 + j
                    index_in.append([0, raw, col])
            return np.array(index_in)

        
    def get_right_top_out(self, h, w, flag, patch=0):
        if flag == 1:
            index_out = []
            for i in range(h//2 - 1):
                for j in range(w//2 - i):
                    raw = i
                    col = w//2 + i + 1 + j
                    index_out.append([patch, raw, col])
            return np.array(index_out)
        elif flag == 0:
            index_out = []
            for i in range(h//2 - 1):
                for j in range(w//2 - 1 - i):
                    raw = i
                    col = w//2 + i + j + 1
                    index_out.append([patch, raw, col])
            return np.array(index_out)
    
    def get_right_top_in_1(self, h, w, flag):
        if flag == 1:
            index_in = []
            for i in range(h//2 - 1):
                for j in range(w//2 - i):
                    raw = h//2 + i
                    col = i + j
                    index_in.append([2, raw, col])
            return np.array(index_in)
        elif flag == 0:
            index_in = []
            for i in range(h//2 - 1):
                for j in range(w//2 - 1 - i):
                    raw = h//2 + i + 1
                    col = i + j + 1
                    index_in.append([2, raw, col])
            return np.array(index_in)
    
    def get_left_bottom_out(self, h, w, flag, patch=0):
        if flag == 1:
            index_out = []
            for i in range(h//2 - 1):
                for j in range(i + 1):
                    raw = h//2 + 1 + i
                    col = j
                    index_out.append([patch, raw, col])
            return np.array(index_out)
        elif flag == 0:
            index_out = []
            for i in range(h//2 - 1):
                for j in range(i + 1):
                    raw = h//2 + 1 + i
                    col = j
                    index_out.append([patch, raw, col])
            return np.array(index_out)
        
    def get_left_bottom_in_0(self, h, w, flag):
        if flag == 1:
            index_in = []     
            for i in range(h//2-1):
                for j in range(i + 1):
                    raw = i
                    col = w//2 + j
                    index_in.append([9, raw, col])
            return np.array(index_in)
        elif flag == 0:
            index_in = []     
            for i in range(h//2-1):
                for j in range(i + 1):
                    raw = i
                    col = w//2 + j
                    index_in.append([9, raw, col])
            return np.array(index_in)
            

    def get_left_top_in_0(self, h, w, flag):
        if flag == 1:
            index_in = []
            for j in range(w//2 + 1):
                for i in range(h//2 - 1 - j):
                    raw = h//2 - 2 - i
                    col = w//2 + j
                    index_in.append([8, raw, col])
            return np.array(index_in)
        elif flag == 0:
            index_in = []
            for j in range(w//2 - 1):
                for i in range(h//2 - 1 - j):
                    raw = h//2 - 2 - i
                    col = w//2 + j
                    index_in.append([8, raw, col])
            return np.array(index_in)
        
    def get_left_top_out(self, h, w, flag, patch=0):
        if flag == 1:
            index_out = []
            for i in range(h//2 - 1):
                for j in range(w//2 - i):
                    raw = i
                    col = j
                    index_out.append([patch, raw, col])
            return np.array(index_out)
        
        elif flag == 0:
            index_out = []
            for i in range(h//2 - 1):
                for j in range(w//2 - 1 - i):
                    raw = i
                    col = j
                    index_out.append([patch, raw, col])
            return np.array(index_out)
    
    def get_right_top_in_0(self, h, w, flag):
        if flag == 1:
            index_in = []
            for j in range(w//2):
                for i in range(h//2 - 1 - j):
                    raw = i + j
                    col = w//2 -j
                    index_in.append([2, raw, col])
            return np.array(index_in)
        elif flag == 0:
            index_in = []
            for j in range(w//2 - 1):
                for i in range(h//2 - 1 - j):
                    raw = i + j 
                    col = w//2 - j - 1
                    index_in.append([2, raw, col])
            return np.array(index_in)
        
    def get_left_bottom_in_1(self, h, w, flag):
        if flag == 1:
            index_in = []
            for j in range(w//2):
                for i in range(j+1):
                    raw = h//2 + i
                    col = w - 1 - j
                    index_in.append([9, raw, col])
            return np.array(index_in)
        elif flag == 0:
            index_in = []
            for j in range(w//2 - 1):
                for i in range(j+1):
                    raw = h//2 + i + 1
                    col = w - 2 - j
                    index_in.append([9, raw, col])
            return np.array(index_in)

    def get_right_bottom_in_1(self, h, w, flag):
        if flag == 1:
            index_in = []
            for j in range(w//2):
                for i in range(j+1):
                    raw = h//2 + j - i
                    col = j + 1
                    index_in.append([3, raw, col])
            return np.array(index_in)
        elif flag == 0:
            index_in = []
            for j in range(w//2 - 1):
                for i in range(j+1):
                    raw = h//2 + j - i + 1
                    col = j + 1
                    index_in.append([3, raw, col])
            return np.array(index_in)
                    
    def get_level_padding_idx(self, level, flag = 0):

        h = 2**(level+1)
        w = 2**(level + 1) - 1

        index0_out = self.get_right_bottom_out(h, w, 1)
        index0_in = self.get_right_bottom_in_0(h, w, 1)

        index1_out = self.get_left_bottom_out(h, w, 1)
        index1_in = self.get_left_bottom_in_0(h, w, 1)

        index2_out = self.get_left_top_out(h, w, 1, patch=1)
        index2_in = self.get_left_top_in_1(h,w, 1)

        index3_out = self.get_right_top_out(h, w, 1, patch=1)
        index3_in = self.get_right_top_in_1(h, w, 1)

        index4_out = self.get_left_top_out(h, w, 1)
        index4_in = self.get_left_top_in_0(h, w, 1)

        index5_out = self.get_right_top_out(h, w, 1)
        index5_in = self.get_right_top_in_0(h, w, 1)

        index6_out = self.get_left_bottom_out(h, w, 1, patch=1)
        index6_in = self.get_left_bottom_in_1(h, w, 1)

        index7_out = self.get_right_bottom_out(h, w, 1, patch=1)
        index7_in = self.get_right_bottom_in_1(h, w, 1) 

            

        index_in = np.concatenate((index0_in, index1_in, index2_in, index3_in, index4_in, index5_in, index6_in, index7_in),axis=0)
        index_out = np.concatenate((index0_out, index1_out, index2_out, index3_out, index4_out, index5_out, index6_out, index7_out),axis=0)

        self.index_lookup[level][0] = torch.Tensor(index_out).type(torch.long).to(self.device)
        self.index_lookup[level][1] =  torch.Tensor(index_in).type(torch.long).to(self.device)

    def get_padding(self, input, level):
        # input_pad = F.pad(input, [1,1,0,0])	#padding / [L,R,U,D]
        bs, c, h, w = input.shape
        input = input.view(bs//10, 10, c, h, w)  # reshape to [bs, 5, c, h, w]
        input_tr = input.permute(0, 1, 3, 4, 2) #change [5,c,h,w] -> [5,h,w,c]
        for i in range(0, 10, 2):
            input_tr[:, (self.index_lookup[level][0][:,0]+i)%10, self.index_lookup[level][0][:,1],  self.index_lookup[level][0][:,2]] = \
            input_tr[:, (self.index_lookup[level][1][:,0]+i)%10, self.index_lookup[level][1][:,1],  self.index_lookup[level][1][:,2]]
        result = input_tr.permute(0,1,4,2,3)
        result = result.view(result.size(0)*10,result.size(2),result.size(3),result.size(4))
        # return result[..., 24:-24,8:-8]  # remove padding
        return result  # remove padding
    
    def return_padding(self, input, level):
        bs, c, h, w = input.shape
        input = input.view(bs//10, 10, c, h, w)  # reshape to [bs, 5, c, h, w]
        input_tr = input.permute(0, 1, 3, 4, 2) #change [5,c,h,w] -> [5,h,w,c]
        for i in range(0, 10, 2):
            # input_tr[:, (self.index_lookup[level][0][:,0]+i)%10, self.index_lookup[level][0][:,1],  self.index_lookup[level][0][:,2]] = \
            # input_tr[:, (self.index_lookup[level][1][:,0]+i)%10, self.index_lookup[level][1][:,1],  self.index_lookup[level][1][:,2]]
            input_tr[:, (self.index_lookup[level][1][:,0]+i)%10, self.index_lookup[level][1][:,1],  self.index_lookup[level][1][:,2]] = \
            input_tr[:, (self.index_lookup[level][0][:,0]+i)%10, self.index_lookup[level][0][:,1],  self.index_lookup[level][0][:,2]] 
            # 0.5 * input_tr[:, (self.index_lookup[level][1][:,0]+i)%10, self.index_lookup[level][1][:,1],  self.index_lookup[level][1][:,2]]
        result = input_tr.permute(0,1,4,2,3)
        result = result.view(result.size(0)*10,result.size(2),result.size(3),result.size(4))
        return result  # remove padding
    
class padding_v4_1(object):
    def __init__(self, list_level):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.index_lookup = {} # 5 128 ； 6 256 ；7 512； 8 1024
        for level in list_level:
            self.index_lookup[level] = {}
            self.get_level_padding_idx(level)

    def get_right_bottom_out(self, h, w, flag, patch=0):
        
            index_out = []
            for i in range(h//2 - 1):
                for j in range(i + 1):
                    raw = h//2 + 1 + i
                    col = w - 1 - i + j 
                    index_out.append([patch, raw, col])
            return np.array(index_out)
        
    def get_right_bottom_out_1(self, h, w, flag, patch=0):
            index_out = []
            for j in range(w//2 - 1):
                for i in range(h//2 - 1 - j):
                    raw = h//2 + 1 + i + j
                    col = w - 1 - j 
                    index_out.append([patch, raw, col])
            return np.array(index_out)
    
    def get_right_bottom_in_0(self, h, w, flag):
            index_in = []
            for i in range(h//2 - 1):
                for j in range(i + 1):
                    raw = i
                    col = w//2 - 1 - i + j 
                    index_in.append([1, raw, col])
            return np.array(index_in)

    def get_left_top_in_1(self, h, w, flag):

            index_in = []
            for i in range(h//2 - 1):
                for j in range(w//2 - 1 - i):
                    raw = h//2 + 1 + i
                    col = w//2 + j
                    index_in.append([0, raw, col])
            return np.array(index_in)

        
    def get_right_top_out(self, h, w, flag, patch=0):

            index_out = []
            for i in range(h//2 - 1):
                for j in range(w//2 - 1 - i):
                    raw = i
                    col = w//2 + i + j + 1
                    index_out.append([patch, raw, col])
            return np.array(index_out)
        
    def get_right_top_out_0(self, h, w, flag, patch=0):
            
            index_out = []
            for j in range(w//2 - 1):
                for i in range(h//2 - 1 - j):
                    raw = h//2 - 1 - i - j
                    col = w - 1 - j
                    index_out.append([patch, raw, col])
            return np.array(index_out)
    
    def get_right_top_in_1(self, h, w, flag):

            index_in = []
            for i in range(h//2 - 1):
                for j in range(w//2 - 1 - i):
                    raw = h//2 + i + 1
                    col = i + j + 1
                    index_in.append([2, raw, col])
            return np.array(index_in)
    
    def get_left_bottom_out(self, h, w, flag, patch=0):
            
            index_out = []
            for i in range(h//2 - 1):
                for j in range(i + 1):
                    raw = h//2 + 1 + i
                    col = j
                    index_out.append([patch, raw, col])
            return np.array(index_out)
        
    def get_left_bottom_out_1(self, h, w, flag, patch=0):

            index_out = []
            for j in range(w//2 - 1):
                for i in range(h//2 - 1 - j):
                    raw = h//2 + 1 + i + j
                    col = j
                    index_out.append([patch, raw, col])
            return np.array(index_out)
        
    def get_left_bottom_in_0(self, h, w, flag):
            
            index_in = []     
            for i in range(h//2-1):
                for j in range(i + 1):
                    raw = i
                    col = w//2 + j
                    index_in.append([9, raw, col])
            return np.array(index_in)
            

    def get_left_top_in_0(self, h, w, flag):

            index_in = []
            i, j = 0, 0
            while j < w // 2 - 1:
                count_j = j
                i = 0
                while i < h // 2 - 1 - j:

                    raw_0 = h//2 - 1 - i//2 - j 
                    col_0 = w - 1 - count_j - i//2  

                    raw_1 = h//2 - 1 - i//2 - j
                    col_1 = w - 1 - count_j - i//2 - 1


                    index_in.append([8, raw_0, col_0])
                    i += 1
                    if i+1 < h // 2 - 1 - j:
                        index_in.append([8, raw_1, col_1])
                        i += 1
                    count_j += 2
                j += 1

            return np.array(index_in)
        
    def get_left_top_out_0(self, h, w, flag, patch=0):

            index_out = []
            for j in range(w//2 - 1):
                for i in range(h//2 - 1 - j):
                    raw = h//2 - 1 - i - j
                    col = j #w//2 - 1 - i - j
                    index_out.append([patch, raw, col])

            return np.array(index_out)
        
    def get_left_top_out(self, h, w, flag, patch=0):

            index_out = []
            for i in range(h//2 - 1):
                for j in range(w//2 - 1 - i):
                    raw = i
                    col = j
                    index_out.append([patch, raw, col])
            return np.array(index_out)
    
    def get_right_top_in_0(self, h, w, flag):

            index_in = []
            i, j = 0, 0
            while j < w // 2 - 1:
                count_j = j
                i = 0
                while i < h // 2 - 1 - j:

                    raw_0 =  h//2 - 1 - i//2 - j 
                    col_0 =  count_j + i//2  

                    raw_1 =  h//2 - 1 - i//2 - j 
                    col_1 =  count_j + i//2 + 1


                    index_in.append([2, raw_0, col_0])
                    i += 1
                    if i+1 < h // 2 - 1 - j:
                        index_in.append([2, raw_1, col_1])
                        i += 1
                    count_j += 2
                j += 1

            return np.array(index_in)
        
        
    def get_left_bottom_in_1(self, h, w, flag):
            index_in = []
            i, j = 0, 0
            while j < w // 2 - 1:
                count_j = j
                i = 0
                while i < h // 2 - 1 - j:

                    raw_0 = h//2 + i//2 + j 
                    col_0 = w - 1 - count_j - i//2  

                    raw_1 = h//2 + i//2 + j
                    col_1 = w - 1 - count_j - i//2 - 1

                    index_in.append([9, raw_0, col_0])
                    i += 1
                    if i+1 < h // 2 - 1 - j:
                        index_in.append([9, raw_1, col_1])
                        i += 1
                    if j == 0:
                        print("raw_0, col_0, raw_1, col_1", i, raw_0, col_0, raw_1, col_1)
                    count_j += 2

                j += 1

            return np.array(index_in)

    def get_right_bottom_in_1(self, h, w, flag):
            index_in = []
            i, j = 0, 0
            while j < w // 2 - 1:
                count_j = j
                i = 0
                while i < h // 2 - 1 - j:

                    raw_0 = h//2 + i//2 + j 
                    col_0 = count_j + i//2  

                    raw_1 = h//2 + i//2 + j
                    col_1 = count_j + i//2 + 1

                    index_in.append([3, raw_0, col_0])
                    i += 1
                    if i+1 < h // 2 - 1 - j:
                        index_in.append([3, raw_1, col_1])
                        i += 1
                    if j == 0:
                        print("raw_0, col_0, raw_1, col_1", i, raw_0, col_0, raw_1, col_1)
                    count_j += 2

                j += 1
            return np.array(index_in)
                    
    def get_level_padding_idx(self, level, flag = 0):

        h = 2**(level+1)
        w = 2**(level + 1)

        index0_out = self.get_right_bottom_out(h, w, 0)
        index0_in = self.get_right_bottom_in_0(h, w, 0)

        index1_out = self.get_left_bottom_out(h, w, 0)
        index1_in = self.get_left_bottom_in_0(h, w, 0)

        index2_out = self.get_left_top_out(h, w, 0, patch=1)
        index2_in = self.get_left_top_in_1(h,w, 0)

        index3_out = self.get_right_top_out(h, w, 0, patch=1)
        index3_in = self.get_right_top_in_1(h, w, 0)

        index4_out = self.get_left_top_out_0(h, w, 0)
        index4_in = self.get_left_top_in_0(h, w, 0)

        index5_out = self.get_right_top_out_0(h, w, 0)
        index5_in = self.get_right_top_in_0(h, w, 0)

        index6_out = self.get_left_bottom_out_1(h, w, 0, patch=1)
        index6_in = self.get_left_bottom_in_1(h, w, 0)

        index7_out = self.get_right_bottom_out_1(h, w, 0, patch=1)
        index7_in = self.get_right_bottom_in_1(h, w, 0) 
            

        index_in = np.concatenate((index0_in, index1_in, index2_in, index3_in, index4_in, index5_in, index6_in, index7_in),axis=0)
        index_out = np.concatenate((index0_out, index1_out, index2_out, index3_out, index4_out, index5_out, index6_out, index7_out),axis=0)

        self.index_lookup[level][0] = torch.Tensor(index_out).type(torch.long).to(self.device)
        self.index_lookup[level][1] =  torch.Tensor(index_in).type(torch.long).to(self.device)

    def get_padding(self, input, level):
        # input_pad = F.pad(input, [1,1,0,0])	#padding / [L,R,U,D]
        bs, c, h, w = input.shape
        input = input.view(bs//10, 10, c, h, w)  # reshape to [bs, 5, c, h, w]
        input_tr = input.permute(0, 1, 3, 4, 2) #change [5,c,h,w] -> [5,h,w,c]
        for i in range(0, 10, 2):
            input_tr[:, (self.index_lookup[level][0][:,0]+i)%10, self.index_lookup[level][0][:,1],  self.index_lookup[level][0][:,2]] = \
            input_tr[:, (self.index_lookup[level][1][:,0]+i)%10, self.index_lookup[level][1][:,1],  self.index_lookup[level][1][:,2]]
        result = input_tr.permute(0,1,4,2,3)
        result = result.view(result.size(0)*10,result.size(2),result.size(3),result.size(4))
        # return result[..., 24:-24,8:-8]  # remove padding
        return result  # remove padding
    
    def return_padding(self, input, level):
        bs, c, h, w = input.shape
        input = input.view(bs//10, 10, c, h, w)  # reshape to [bs, 5, c, h, w]
        input_tr = input.permute(0, 1, 3, 4, 2) #change [5,c,h,w] -> [5,h,w,c]
        for i in range(0, 10, 2):
            # input_tr[:, (self.index_lookup[level][0][:,0]+i)%10, self.index_lookup[level][0][:,1],  self.index_lookup[level][0][:,2]] = \
            # input_tr[:, (self.index_lookup[level][1][:,0]+i)%10, self.index_lookup[level][1][:,1],  self.index_lookup[level][1][:,2]]
            input_tr[:, (self.index_lookup[level][1][:,0]+i)%10, self.index_lookup[level][1][:,1],  self.index_lookup[level][1][:,2]] = \
            input_tr[:, (self.index_lookup[level][0][:,0]+i)%10, self.index_lookup[level][0][:,1],  self.index_lookup[level][0][:,2]] 
            # 0.5 * input_tr[:, (self.index_lookup[level][1][:,0]+i)%10, self.index_lookup[level][1][:,1],  self.index_lookup[level][1][:,2]]
        result = input_tr.permute(0,1,4,2,3)
        result = result.view(result.size(0)*10,result.size(2),result.size(3),result.size(4))
        return result  # remove padding
    
class padding_v4_0(object):
    def __init__(self, list_level):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.index_lookup = {} # 5 128 ； 6 256 ；7 512； 8 1024
        for level in list_level:
            self.index_lookup[level] = {}
            self.get_level_padding_idx(level)

    def get_right_bottom_out(self, h, w, flag, patch=0):
        
            index_out = []
            for i in range(h//2 - 1):
                for j in range(i + 1):
                    raw = h//2 + 1 + i
                    col = w - 1 - i + j 
                    index_out.append([patch, raw, col])
            return np.array(index_out)
        
    def get_right_bottom_out_1(self, h, w, flag, patch=0):
            index_out = []
            for j in range(w//2 - 1):
                for i in range(h//2 - 1 - j):
                    raw = h//2 + 1 + i + j
                    col = w - 1 - j 
                    index_out.append([patch, raw, col])
            return np.array(index_out)
    
    def get_right_bottom_in_0(self, h, w, flag):
            index_in = []
            for i in range(h//2 - 1):
                for j in range(i + 1):
                    raw = i
                    col = w//2 - 1 - i + j 
                    index_in.append([1, raw, col])
            return np.array(index_in)

    def get_left_top_in_1(self, h, w, flag):

            index_in = []
            for i in range(h//2 - 1):
                for j in range(w//2 - 1 - i):
                    raw = h//2 + 1 + i
                    col = w//2 + j
                    index_in.append([0, raw, col])
            return np.array(index_in)

        
    def get_right_top_out(self, h, w, flag, patch=0):

            index_out = []
            for i in range(h//2 - 1):
                for j in range(w//2 - 1 - i):
                    raw = i
                    col = w//2 + i + j + 1
                    index_out.append([patch, raw, col])
            return np.array(index_out)
        
    def get_right_top_out_0(self, h, w, flag, patch=0):
            
            index_out = []
            for j in range(w//2 - 1):
                for i in range(h//2 - 1 - j):
                    raw = h//2 - 1 - i - j
                    col = w - 1 - j
                    index_out.append([patch, raw, col])
            return np.array(index_out)
    
    def get_right_top_in_1(self, h, w, flag):

            index_in = []
            for i in range(h//2 - 1):
                for j in range(w//2 - 1 - i):
                    raw = h//2 + i + 1
                    col = i + j + 1
                    index_in.append([2, raw, col])
            return np.array(index_in)
    
    def get_left_bottom_out(self, h, w, flag, patch=0):
            
            index_out = []
            for i in range(h//2 - 1):
                for j in range(i + 1):
                    raw = h//2 + 1 + i
                    col = j
                    index_out.append([patch, raw, col])
            return np.array(index_out)
        
    def get_left_bottom_out_1(self, h, w, flag, patch=0):

            index_out = []
            for j in range(w//2 - 1):
                for i in range(h//2 - 1 - j):
                    raw = h//2 + 1 + i + j
                    col = j
                    index_out.append([patch, raw, col])
            return np.array(index_out)
        
    def get_left_bottom_in_0(self, h, w, flag):
            
            index_in = []     
            for i in range(h//2-1):
                for j in range(i + 1):
                    raw = i
                    col = w//2 + j
                    index_in.append([9, raw, col])
            return np.array(index_in)
            

    def get_left_top_in_0(self, h, w, flag):

            index_in = []
            i, j = 0, 0
            while j < w // 2 - 1:
                count_j = j
                i = 0
                while i < h // 2 - 1 - j:

                    raw_0 = h//2 - 1 - i//2 - j 
                    col_0 = w - 1 - count_j - i//2  

                    raw_1 = h//2 - 1 - i//2 - j
                    col_1 = w - 1 - count_j - i//2 - 1


                    index_in.append([8, raw_0, col_0])
                    i += 1
                    if i+1 < h // 2 - 1 - j:
                        index_in.append([8, raw_1, col_1])
                        i += 1
                    count_j += 2
                j += 1

            return np.array(index_in)
        
    def get_left_top_out_0(self, h, w, flag, patch=0):

            index_out = []
            for j in range(w//2 - 1):
                for i in range(h//2 - 1 - j):
                    raw = h//2 - 1 - i - j
                    col = j #w//2 - 1 - i - j
                    index_out.append([patch, raw, col])

            return np.array(index_out)
        
    def get_left_top_out(self, h, w, flag, patch=0):

            index_out = []
            for i in range(h//2 - 1):
                for j in range(w//2 - 1 - i):
                    raw = i
                    col = j
                    index_out.append([patch, raw, col])
            return np.array(index_out)
    
    def get_right_top_in_0(self, h, w, flag):

            index_in = []
            i, j = 0, 0
            while j < w // 2 - 1:
                count_j = j
                i = 0
                while i < h // 2 - 1 - j:

                    raw_0 =  h//2 - 1 - i//2 - j 
                    col_0 =  count_j + i//2  

                    raw_1 =  h//2 - 1 - i//2 - j 
                    col_1 =  count_j + i//2 + 1


                    index_in.append([2, raw_0, col_0])
                    i += 1
                    if i+1 < h // 2 - 1 - j:
                        index_in.append([2, raw_1, col_1])
                        i += 1
                    count_j += 2
                j += 1

            return np.array(index_in)
        
        
    def get_left_bottom_in_1(self, h, w, flag):
            index_in = []
            i, j = 0, 0
            while j < w // 2 - 1:
                count_j = j
                i = 0
                while i < h // 2 - 1 - j:

                    raw_0 = h//2 + i//2 + j 
                    col_0 = w - 1 - count_j - i//2  

                    raw_1 = h//2 + i//2 + j
                    col_1 = w - 1 - count_j - i//2 - 1

                    index_in.append([9, raw_0, col_0])
                    i += 1
                    if i+1 < h // 2 - 1 - j:
                        index_in.append([9, raw_1, col_1])
                        i += 1
                    if j == 0:
                        print("raw_0, col_0, raw_1, col_1", i, raw_0, col_0, raw_1, col_1)
                    count_j += 2

                j += 1

            return np.array(index_in)

    def get_right_bottom_in_1(self, h, w, flag):
            index_in = []
            i, j = 0, 0
            while j < w // 2 - 1:
                count_j = j
                i = 0
                while i < h // 2 - 1 - j:

                    raw_0 = h//2 + i//2 + j 
                    col_0 = count_j + i//2  

                    raw_1 = h//2 + i//2 + j
                    col_1 = count_j + i//2 + 1

                    index_in.append([3, raw_0, col_0])
                    i += 1
                    if i+1 < h // 2 - 1 - j:
                        index_in.append([3, raw_1, col_1])
                        i += 1
                    if j == 0:
                        print("raw_0, col_0, raw_1, col_1", i, raw_0, col_0, raw_1, col_1)
                    count_j += 2

                j += 1
            return np.array(index_in)
                    
    def get_level_padding_idx(self, level, flag = 0):

        h = 2**(level+1)
        w = 2**(level + 1)

        index0_out = self.get_right_bottom_out(h, w, 0)
        index0_in = self.get_right_bottom_in_0(h, w, 0)

        index1_out = self.get_left_bottom_out(h, w, 0)
        index1_in = self.get_left_bottom_in_0(h, w, 0)

        index2_out = self.get_left_top_out(h, w, 0, patch=1)
        index2_in = self.get_left_top_in_1(h,w, 0)

        index3_out = self.get_right_top_out(h, w, 0, patch=1)
        index3_in = self.get_right_top_in_1(h, w, 0)

        index4_out = self.get_left_top_out_0(h, w, 0)
        index4_in = self.get_left_top_in_0(h, w, 0)

        index5_out = self.get_right_top_out_0(h, w, 0)
        index5_in = self.get_right_top_in_0(h, w, 0)

        index6_out = self.get_left_bottom_out_1(h, w, 0, patch=1)
        index6_in = self.get_left_bottom_in_1(h, w, 0)

        index7_out = self.get_right_bottom_out_1(h, w, 0, patch=1)
        index7_in = self.get_right_bottom_in_1(h, w, 0) 
            

        index_in = np.concatenate((index0_in, index1_in, index2_in, index3_in, index4_in, index5_in, index6_in, index7_in),axis=0)
        index_out = np.concatenate((index0_out, index1_out, index2_out, index3_out, index4_out, index5_out, index6_out, index7_out),axis=0)

        self.index_lookup[level][0] = torch.Tensor(index_out).type(torch.long).to(self.device)
        self.index_lookup[level][1] =  torch.Tensor(index_in).type(torch.long).to(self.device)

    def get_padding(self, input, level):
        # input_pad = F.pad(input, [1,1,0,0])	#padding / [L,R,U,D]
        bs, c, h, w = input.shape
        input = input.view(bs//10, 10, c, h, w)  # reshape to [bs, 5, c, h, w]
        input_tr = input.permute(0, 1, 3, 4, 2) #change [5,c,h,w] -> [5,h,w,c]
        for i in range(0, 10, 2):
            input_tr[:, (self.index_lookup[level][0][:,0]+i)%10, self.index_lookup[level][0][:,1],  self.index_lookup[level][0][:,2]] = \
            input_tr[:, (self.index_lookup[level][1][:,0]+i)%10, self.index_lookup[level][1][:,1],  self.index_lookup[level][1][:,2]]
        result = input_tr.permute(0,1,4,2,3)
        result = result.view(result.size(0)*10,result.size(2),result.size(3),result.size(4))
        # return result[..., 24:-24,8:-8]  # remove padding
        return result  # remove padding
    
    def return_padding(self, input, level):
        bs, c, h, w = input.shape
        input = input.view(bs//10, 10, c, h, w)  # reshape to [bs, 5, c, h, w]
        input_tr = input.permute(0, 1, 3, 4, 2) #change [5,c,h,w] -> [5,h,w,c]
        for i in range(0, 10, 2):
            # input_tr[:, (self.index_lookup[level][0][:,0]+i)%10, self.index_lookup[level][0][:,1],  self.index_lookup[level][0][:,2]] = \
            # input_tr[:, (self.index_lookup[level][1][:,0]+i)%10, self.index_lookup[level][1][:,1],  self.index_lookup[level][1][:,2]]
            input_tr[:, (self.index_lookup[level][1][:,0]+i)%10, self.index_lookup[level][1][:,1],  self.index_lookup[level][1][:,2]] = \
            input_tr[:, (self.index_lookup[level][0][:,0]+i)%10, self.index_lookup[level][0][:,1],  self.index_lookup[level][0][:,2]] 
            # 0.5 * input_tr[:, (self.index_lookup[level][1][:,0]+i)%10, self.index_lookup[level][1][:,1],  self.index_lookup[level][1][:,2]]
        result = input_tr.permute(0,1,4,2,3)
        result = result.view(result.size(0)*10,result.size(2),result.size(3),result.size(4))
        return result  # remove padding
if __name__ == "__main__":
    level = 3 
    h, w = 4*(2**level), 2**level
    pad_old = padding_old(level)
    pad_new = padding()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = torch.randn(100*5*h*w).view(100,5,1,h,w).to(device)  #sub_img_num, feature_num, height, width
    result = pad_old.get_padding(input)
    result2 = pad_new.get_padding(input, level)
    if ((~(result==result2)).sum())==0:
        print('same result')
    else:
        print('different result')
    import pdb; pdb.set_trace()

