
import torch
import numpy as np
import torch.nn.functional as F

class padding(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.index_lookup = {}
        for level in range(1, 8):
            self.index_lookup[level] = {}
            self.get_level_padding_idx(level)

    # def get_level_padding_idx(self, level, lr=8, ud=8):
    #     h, w = 4*(2**level), 2**level
    #     index1_in, index2_in, index3_in, index4_in, index5_in, index6_in = \
    #         [], [], [], [], [], [] #pink, brown, purple, green, yellow, blue
    #     index1_out, index2_out, index3_out, index4_out, index5_out, index6_out = \
    #         [], [], [], [], [], []

    #     for i in range(h//2):
    #         index1_in.append([4,h//2-1-i +2,w-1 + 1])
    #     index1_in.append([3,0+2,w-1+1])

    #     for i in range(h//2+2):
    #         if i == 0:
    #             continue
    #         else :
    #             index1_out.append([0, (i+1)%2, i//2 ])

    #     for i in range(h//2 + 1):
    #         index2_in.append([4,h -1 - i+2,w-1+1])

    #     for i in range(h//2 + 1):
    #         index2_out.append([0, (h//2 + 2) - 1 - i, 0])

    #     index3_in.append([3,h-1 +2,0+1])


    #     for i in range(w-1):
    #         index3_in.append([4, h-1+2, i+1 ])
    #         index3_in.append([4, h-2+2, i+1 ])
    #     index3_in.append([4, h-1+2, w-1+1])

    #     for i in range(h//2):
    #         index3_out.append([0, (h + 2) -1 - i, 0])

    #     for i in range(h//2):
    #         index4_in.append([1,h+3 - (h//2-1-i +2), w+ 1 -(w-1 + 1)])
    #     index4_in.append([2,h+3 -(0+2),w+1 - (w-1+1)])

    #     for i in range(h//2+2):
    #         if i == 0:
    #             continue
    #         else :
    #             index4_out.append([0, h+3 - ((i+1)%2), w+1 - (i//2) ])


    #     for i in range(h//2 + 1):
    #         index5_in.append([1,h+3 - (h -1 - i+2),w+1 - (w-1+1)])

    #     for i in range(h//2 + 1):
    #         index5_out.append([0,h+3 - ((h//2 + 2) - 1 - i), w+1 - 0])


    #     index6_in.append([2,h+3 - (h-1+2),w+1 - (0+1)])

    #     for i in range(w-1):
    #         index6_in.append([1, h+3 - (h-1+2), w+1 - (i+1) ])
    #         index6_in.append([1, h+3 - (h-2+2), w+1 - (i+1) ])
    #     index6_in.append([1, h+3 - (h-1+2), w+1 - (w-1+1)])

    #     for i in range(h//2):
    #         index6_out.append([0, h+3 - ((h + 2) -1 - i), w+1 - 0])

    #     index1_in = np.array(index1_in)
    #     index2_in = np.array(index2_in)
    #     index3_in = np.array(index3_in)
    #     index4_in = np.array(index4_in)
    #     index5_in = np.array(index5_in)
    #     index6_in = np.array(index6_in)

    #     index1_out = np.array(index1_out)
    #     index2_out = np.array(index2_out)
    #     index3_out = np.array(index3_out)
    #     index4_out = np.array(index4_out)
    #     index5_out = np.array(index5_out)
    #     index6_out = np.array(index6_out)

    #     index_in = np.concatenate((index1_in, index2_in, index3_in, index4_in, index5_in, index6_in),axis=0)
    #     index_out = np.concatenate((index1_out, index2_out, index3_out, index4_out, index5_out, index6_out),axis=0)

    #     self.index_lookup[level][0] = torch.Tensor(index_out).type(torch.long).to(self.device)
    #     self.index_lookup[level][1] =  torch.Tensor(index_in).type(torch.long).to(self.device)
    def get_level_padding_idx(self, level, lr=8, ud=16):
        """
        计算给定level和padding尺寸下的输入和输出索引。
        lr: left/right padding size
        ud: up/down padding size
        """
        h, w = 4*(2**level), 2**level
        index1_in, index2_in, index3_in, index4_in, index5_in, index6_in = \
            [], [], [], [], [], [] #pink, brown, purple, green, yellow, blue
        index1_out, index2_out, index3_out, index4_out, index5_out, index6_out = \
            [], [], [], [], [], []

        # 填充后的高度和宽度相关的末端索引参考点
        # 原始代码中的 h+3 对应 (h + 2*旧ud) - 1，其中旧ud=2
        # 原始代码中的 w+1 对应 (w + 2*旧lr) - 1，其中旧lr=1
        hp_far_end = h + 2*ud -1
        wp_far_end = w + 2*lr -1

        # --- Block 1 (Pink) ---
        for i in range(h//2):
            # 原来是 +2 (ud_old), +1 (lr_old)
            index1_in.append([4, h//2-1-i + ud, w-1 + lr])
        index1_in.append([3, 0 + ud, w-1 + lr])

        for i in range(h//2+2):
            if i == 0:
                continue
            else :
                # 假设 (i+1)%2 和 i//2 是相对于内容区域左上角的偏移
                index1_out.append([0, (i+1)%2 + ud, i//2 + lr])

        # --- Block 2 (Brown) ---
        for i in range(h//2 + 1):
            index2_in.append([4, h -1 - i + ud, w-1 + lr])

        for i in range(h//2 + 1):
            # 假设 (h//2 + 2) - 1 - i 是内容区域的行偏移，0是列偏移
            index2_out.append([0, (h//2 + 2) - 1 - i + ud, 0 + lr])

        # --- Block 3 (Purple) ---
        index3_in.append([3, h-1 + ud, 0 + lr])
        for i in range(w-1): # 注意这里的i是内容区域的列偏移
            index3_in.append([4, h-1 + ud, i + lr ])
            index3_in.append([4, h-2 + ud, i + lr ])
        index3_in.append([4, h-1+ud, w-1+lr]) # w-1是内容区域最后一列

        for i in range(h//2):
            index3_out.append([0, (h + 2) -1 - i + ud, 0 + lr])


        # --- Block 4 (Green) ---
        for i in range(h//2):
            # y_coord_in_padded_space = h//2-1-i + ud
            # x_coord_in_padded_space = w-1 + lr
            index4_in.append([1, hp_far_end - (h//2-1-i + ud), wp_far_end - (w-1 + lr)])
        # y_coord_in_padded_space_origin = 0 + ud
        index4_in.append([2, hp_far_end - (0+ud), wp_far_end - (w-1+lr)])

        for i in range(h//2+2):
            if i == 0:
                continue
            else :
                # ((i+1)%2) 和 (i//2) 是小偏移量，从远端减去
                index4_out.append([0, hp_far_end - ((i+1)%2), wp_far_end - (i//2) ])

        # --- Block 5 (Yellow) ---
        for i in range(h//2 + 1):
            # y_coord = h -1 - i+ud
            # x_coord = w-1+lr
            index5_in.append([1, hp_far_end - (h -1 - i+ud), wp_far_end - (w-1+lr)])

        for i in range(h//2 + 1):
            # ((h//2 + 2) - 1 - i) 是一个偏移量
            index5_out.append([0, hp_far_end - ((h//2 + 2) - 1 - i), wp_far_end - (0 + lr)]) # 修正：原为 w+1 - 0，应为 wp_far_end - (0+lr) 或 wp_far_end - lr

        # --- Block 6 (Blue) ---
        # y_coord1 = h-1+ud
        # x_coord1 = 0+lr
        index6_in.append([2, hp_far_end - (h-1+ud), wp_far_end - (0+lr)])

        for i in range(w-1): # i是内容区域的列偏移
            # y_coord2 = h-1+ud
            # x_coord_loop = i+lr
            index6_in.append([1, hp_far_end - (h-1+ud), wp_far_end - (i+lr) ])
            # y_coord3 = h-2+ud
            index6_in.append([1, hp_far_end - (h-2+ud), wp_far_end - (i+lr) ])
        # y_coord4 = h-1+ud
        # x_coord_end = w-1+lr
        index6_in.append([1, hp_far_end - (h-1+ud), wp_far_end - (w-1+lr)])

        for i in range(h//2):
            # ((h + 2) -1 - i) 是一个偏移量
            index6_out.append([0, hp_far_end - ((h + 2) -1 - i), wp_far_end - (0 + lr)]) # 修正：原为 w+1 - 0

        # ... (numpy array conversion and concatenation remain the same)
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

        # 确保为当前level和padding组合存储索引
        # 如果lr/ud会变化，可能需要更复杂的key，例如 (level, lr, ud)
        if level not in self.index_lookup:
            self.index_lookup[level] = {}
        # 假设对于一个level，lr和ud是固定的，或者在get_padding中动态生成/获取
        self.index_lookup[level]['in'] = torch.Tensor(index_in).type(torch.long).to(self.device)
        self.index_lookup[level]['out'] = torch.Tensor(index_out).type(torch.long).to(self.device)


    # def get_padding(self, input, level):
    #     input = input.reshape(input.size(0)//5, 5, input.size(1), input.size(2), input.size(3))
    #     input_pad = F.pad(input, [1,1,2,2])	#padding / [L,R,U,D]
    #     input_pad_tr = input_pad.permute(0, 1, 3, 4, 2) #change [5,c,h,w] -> [5,h,w,c]
    #     for i in range(5):
    #         input_pad_tr[:, (self.index_lookup[level][0][:,0]+i)%5, \
    #                          self.index_lookup[level][0][:,1], \
    #                          self.index_lookup[level][0][:,2]] = \
    #         input_pad_tr[:, (self.index_lookup[level][1][:,0]+i)%5, \
    #                          self.index_lookup[level][1][:,1], \
    #                          self.index_lookup[level][1][:,2]]
    #     result = input_pad_tr.permute(0,1,4,2,3)
    #     result = result.view(result.size(0)*5,result.size(2),result.size(3),result.size(4))
    #     return result
    def get_padding(self, input_tensor, level, lr=8, ud=16): # 使用 input_tensor 避免与内置 `input`冲突
        """
        对输入张量进行填充，并根据预计算的索引进行边界处理。
        lr: left/right padding size, 默认为 8
        ud: up/down padding size, 默认为 16
        """
        # 确保索引已为当前level和padding尺寸生成
        # 注意：如果每次调用get_padding时lr/ud都可能不同，
        # 那么get_level_padding_idx应该在这里被调用，或者index_lookup使用更复杂的键。
        # 为简单起见，这里假设在某个初始化阶段已经为特定的lr/ud调用了get_level_padding_idx。
        # 或者，你可以在这里检查并按需生成：
        # key_tuple = (level, lr, ud) # 如果需要更灵活的缓存
        # if level not in self.index_lookup or \
        #    not torch.all(self.index_lookup[level]['in'][0,1:3] == torch.tensor([h//2-1-0+ud, w-1+lr], device=self.device, dtype=torch.long)): # 粗略检查是否为当前padding生成
        #     print(f"Warning: Indices for level {level} might not match lr={lr}, ud={ud}. Consider re-generating.")
            # 或者直接在这里调用: self.get_level_padding_idx(level, lr, ud)

        input_tensor = input_tensor.reshape(input_tensor.size(0)//5, 5, input_tensor.size(1), input_tensor.size(2), input_tensor.size(3))
        b_div_5, _, c, h_orig, w_orig = input_tensor.shape # 假设输入已经是 (B/5, 5, C, H, W)
                                                       # 或者如下面的reshape所示
        
        # 如果输入是 (B, C, H, W)
        # input_tensor = input_tensor.reshape(input_tensor.size(0)//5, 5, input_tensor.size(1), input_tensor.size(2), input_tensor.size(3))
        # 如果已经是 (B_actual, 5, C, H, W)
        num_batches = input_tensor.size(0)

        # padding / [L,R,U,D] -> lr, lr, ud, ud
        input_pad = F.pad(input_tensor, [lr, lr, ud, ud])
        # input_pad shape: (num_batches, 5, C, H_padded, W_padded)

        input_pad_tr = input_pad.permute(0, 1, 3, 4, 2) #change [N,5,C,H_pad,W_pad] -> [N,5,H_pad,W_pad,C]

        # 获取预计算的索引
        # 确保使用正确的键，如果 get_level_padding_idx 中修改了存储方式
        idx_out_coords = self.index_lookup[level]['out']
        idx_in_coords = self.index_lookup[level]['in']

        for i in range(5): # 对于5个面中的每一个（这部分逻辑保持不变）
            # N, face_idx_dst, row_dst, col_dst, C
            dest_face_indices = (idx_out_coords[:,0]+i)%5
            dest_row_indices = idx_out_coords[:,1]
            dest_col_indices = idx_out_coords[:,2]

            # N, face_idx_src, row_src, col_src, C
            src_face_indices = (idx_in_coords[:,0]+i)%5
            src_row_indices = idx_in_coords[:,1]
            src_col_indices = idx_in_coords[:,2]
            
            input_pad_tr[:, dest_face_indices, dest_row_indices, dest_col_indices] = \
                input_pad_tr[:, src_face_indices, src_row_indices, src_col_indices]

        result = input_pad_tr.permute(0,1,4,2,3) # [N,5,C,H_pad,W_pad]
        result = result.view(result.size(0)*5,result.size(2),result.size(3),result.size(4)) # 如果需要展平批次和面
        return result
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

