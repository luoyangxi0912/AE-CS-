# # # -*- coding: utf-8 -*-
# # """
# # Created on Thu Sep  8 23:33:33 2022
# #
# # @author: Joff Pan
# # """
# import os
# import sys
#
import torch
import numpy as np
from joff._run._runner import Runner
from joff._load._make_dataset import _make_dataset
from torch.utils.data import dataset,DataLoader
from joff._run._epoch import batch_testing
from joff._test._fd import offline_modeling,_get_mm,_get_ts, online_monitoring
import matplotlib.pyplot as plt
import sys
import time
import torch
from joff._func._dvc import _2dvc
from joff._func._np import _2np
from functorch import vmap, jacrev, jacfwd, hessian
# from joff._nn._vmap import _get_jcb_hess
def _get_jcb_hess(self):
    _2dvc(self, self.dvc)
    if self.training: self.eval()

    loader = self.unshuf_train_loader if hasattr(self,'unshuf_train_loader') else self.train_loader
    I = _2dvc(torch.eye(loader.X.shape[1]), self.dvc)
    M, J, H = [], [], []

    print()
    start = time.perf_counter()
    with torch.no_grad():
        if hasattr(self, 'mv_normal'):
            RD_Test = _2dvc(self.mv_normal.sample(torch.Size([loader.X.shape[0]])), self.dvc)
            p = 0
        for b_id, (x, l) in enumerate(loader):
            x, l = x.to(self.dvc), l.to(self.dvc)
            self._label = l

            if hasattr(self, 'mv_normal'):
                # self.mm = 'lv'
                _rd_test = RD_Test[p: p + x.size(0)]
                p += x.size(0)
                y = self.forward_for_vmap(x, _rd_test)
                jacobian = vmap(jacrev(self.forward_for_vmap, argnums=0))(x, _rd_test)
                hess = vmap(hessian(self.forward_for_vmap, argnums=0))(x, _rd_test)
            else:
                y = self.forward(x)
                jacobian = vmap(jacrev(self.forward))(x)
                hess = vmap(hessian(self.forward))(x)

            if (b_id + 1) % 10 == 0 or (b_id + 1) == len(loader):
                msg_str = 'Calculating batch gradients: {}/{}'.format(b_id + 1, len(loader))
                sys.stdout.write('\r' + msg_str + '                                    ')
                sys.stdout.flush()
            if self.mm == 're':
                jacobian -= I
                # mm = y
                mm = x - y
            else:
                mm = y
            M.append(_2np(mm))
            J.append(_2np(jacobian))
            H.append(_2np(hess))
    M, J, H = np.concatenate(M, 0), np.concatenate(J, 0), np.concatenate(H, 0)
    M_size, J_size, H_size = M.size / M.shape[0], J.size / J.shape[0], H.size / H.shape[0]
    self.Jacob = J


if __name__ == '__main__':
    def _get_T2_online(model,X,Y):
        MM = X-Y
        if model.if_minus_mean:
            MM -= model._MM_mean
        T2_cov_MMt = np.linalg.lstsq(model._T2_cov, MM.T, rcond=None)[0].T
        T2 = (MM[:, np.newaxis, :] @ T2_cov_MMt[:, :, np.newaxis]).reshape(-1, )
        return T2,MM


    torch.manual_seed(12)
    # np.random.seed(12)

    from joff._load._read_hy_data import _load_hy_data
    D = _load_hy_data()

    dynamic=18
    p = {'models': ['DAE','ND_DAE','VAE'],
         # 'structs': [[10,'*10','/2'],
         #             [10, '*10', '/2', '/2']],
         # 'structs': [[33 * dynamic, '*10', '/10'],
         #             [33 * dynamic, '*10', '/2', '/5']],
         'structs': [[52*dynamic  , '/2', '/2'],
                     [52*dynamic , '/2', '/2', '/2']],
         # 'structs': [[5 * dynamic, '*20', '/5'],
         #             [5 * dynamic, '*20', '/2', '/5']],
         'acts': [(1, ['a'], ['a']), #最后一个是输出层 前面的是隐层循环
                  (1, ['s'], ['s', 'a']),
                  (1, ['g', 's'], ['s', 'a']),
                  (2, ['s','l', 's'], ['s', 'l', 'a']),
                  (2, ['g', 'g', 's'], ['g', 'a','a']),
                  (2, ['t', 't', 's'], ['t', 's','a']),

                  (1, ['g', 'g'], ['g', 'a']),
                  (1, ['t'], ['t', 'a']),
                  (1, ['q', 'g'], ['q', 'a']),
                  (2, ['q', 't', 'g'], ['g', 't', 'a']),
                  (2, ['s', 't', 's'], ['t', 't', 'a']),
                  (2, ['s', 's', 's'], ['s', 's', 'a'])
                  ],
         # 'acts': [(1, ['a', 'a'], ['g', 'a']),  # 42
         #          (1, ['s', 'a'], ['g', 'a']),  # 12
         #          (1, ['g', 'a'], ['g', 'a']),  # 42
         #          (1, ['q', 'a'], ['g', 'a']),  # 42 12
         #          (2, ['q', 'q', 'a'], ['s', 's', 'a']),  # 333
         #          (2, ['q', 'a', 'a'], ['s', 's', 'a']),  # 42
         #          (2, ['q', 's', 'a'], ['s', 's', 'a']),  # 42
         #          (2, ['s', 's', 'a'], ['s', 's', 'a']),  # 42
         #          (1, ['l'], ['l'])
         #          ],
         #'drop_rate': 0.3,
         'lr': 1e-3,
         'load_datas': [D,
                        {'special': 'CSTR_fd',
                         'scaler': ['st', 'oh']  # 1
                         },
                        {'special': 'TE', 'stack': dynamic, 'scaler': ['st', 'oh']},  # 2
                        {'special': 'TE_fd', 'stack': dynamic, 'scaler': ['st', 'oh']}  # 3
             , {'special': 'Hycrack', 'scaler': ['st', 'oh']}  # 4
             , {'special': 'TE_test', 'stack': dynamic, 'scaler': ['st', 'oh']}  # 5 TE test里面的数据训练+测试
             , {'special': 'TE_zajiao', 'stack': dynamic, 'scaler': ['st', 'oh']}
                        ,{'special': 'TTS', 'stack': dynamic,'scaler': ['st','oh']}
                        ,{'special': 'TE_自己生成的', 'stack': dynamic,'scaler': ['st','oh']}
                        ],
         # 'alf':(2,1)
         '_addi_name': '[unsup_res]'

         }
    # p['if_epoch_eval'] = [False, True]
    # p['mm'] = 'lv'
    p['if_vmap'] = False #jacob
    p['if_simple_msg'] = True
    # R.run(dataset_id = 1, e = 20, run_times = 3
    #           #if_save_plot = False
    #           )

    dataset_id = 1

    module_id = 1
    act_id = 2

    if module_id == 1: m_name = 'DAE'
    elif module_id == 2: m_name = 'ND_DAE'
    elif module_id == 3: m_name = 'VAE'

    for number in range(act_id,act_id+1):
        R = Runner(**p)
        model = R._get_model(module_id = module_id, dataset_id = dataset_id, act_id = number)
        # model.hook_fr_layer()
        model.eval()
        model.run(e = 1, run_times = 1,load_para_type='last',
                  # load_file_name=('../Runner[unsup_res] (d1m2a12)(1)/'+m_name+'_a'+str(number)+' (3 times)/1st/','Last')
                  load_file_name=('../Result/'+m_name+'[unsup_res]_d'+str(dataset_id)+'a'+str(number),'Last')
                  # load_file_name=('../Result/'+m_name+'[vae_lv]_d'+str(dataset_id)+'a'+str(number),'Last')
                  ,ts=['T2'], thrd_meth=['ineq'],mm='re'
                 #if_save_plot = False
                 )
        #读取模型参数
        # model.load_state_dict(torch.load('../Runner[unsup_res] (d1m2a12)(1)/DAE_a1 (3 times)/2nd/Last'))
        #重写test_loader
        trainX, trainY = torch.from_numpy(model.dataset.train_X[0]).float(), torch.from_numpy(model.dataset.train_Y[0]).float()

        testX, testY = torch.from_numpy(np.concatenate(model.dataset.test_X, axis=0)).float(),\
                       torch.from_numpy(np.concatenate(model.dataset.test_Y, axis=0)).float()
        X_concate = torch.cat((trainX,testX), dim=0)
        Y_concate = torch.cat((trainY,testY), dim=0)

        plt_set = dataset.TensorDataset(X_concate, Y_concate)
        model.test_loader = DataLoader(plt_set, batch_size = 16,
                                              shuffle = False, drop_last = False)
        model.test_loader.X = X_concate
        input, output, latent, _  = batch_testing(model,loader = 'test')#array 24020*10 array 24020*10 array 24020*11
        offline_modeling(model,X=input[:trainX.shape[0]],Y=output[:trainX.shape[0]],Z=latent[:trainX.shape[0]],p=model.kwargs)
        T2,thrd = model._TS[0],model._thrd[0,0]

        model.kwargs['if_eval_perf'] = False
        online_monitoring(model, input[trainX.shape[0]:], output[trainX.shape[0]:],
                          latent[trainX.shape[0]:], testY, model.kwargs)

        #
        # input = torch.tensor(input,dtype=torch.float32)
        # output = torch.tensor(output,dtype=torch.float32)
        # latent = torch.tensor(latent,dtype=torch.float32)
        # # hidden1 = torch.tensor(hidden1,dtype=torch.float32)
        # hidden1 = torch.randn(3,4)
        # # model.unshuf_train_loader = model.test_loader
        # # _get_jcb_hess(model)
        # model.Jacob = torch.randn(2,3,4)
        # # model.Jacob 24020*10*10  test_Y0 list:10 train_Y0 list:1
        # # model.dataset.train_Y0.extend(model.dataset.test_Y0)###################
        #
        # normal_index = torch.tensor(np.where(Y_concate.numpy()[:,0]==1),dtype=torch.long).squeeze(0)#对的
        # # normal_index = torch.tensor(np.where(np.concatenate(Y_concate[:,0])==1)[0],dtype=torch.long)#对的
        # fault_index = list(set([i for i in range(input.shape[0])])- set(normal_index.tolist()))
        # n_X,n_Y,n_J,n_L,h1 = input[normal_index],output[normal_index],torch.tensor(model.Jacob,dtype=torch.float32),latent[normal_index],hidden1#,hidden1[normal_index]
        # n_X, n_Y, n_J2,n_L,h1 = torch.sqrt(torch.sum(torch.square(n_X),1)),torch.sqrt(torch.sum(torch.square(n_Y),1)),torch.sqrt(torch.sum(torch.sum(torch.square(n_J),2),1)),torch.sqrt(torch.sum(torch.square(n_L),1)),torch.sqrt(torch.sum(torch.square(h1),1))
        # n_J1 = torch.mean(torch.mean(torch.abs(n_J),2),1)
        # with open('jacob.txt','a') as f:
        #     f.write('E[Jacob_1] of'+m_name+'_a'+str(number)+'='+str(torch.mean(n_J1))+'\n')
        # print('E[Jacob_1] of '+m_name+'_a'+str(number)+'=',torch.mean(n_J1))
        # j2 = n_J2.numpy()
        # j1 = n_J1.numpy()
        # f_X,f_Y,f_L = input[fault_index],output[fault_index],latent[fault_index]#,torch.tensor(model.Jacob[fault_index],dtype=torch.float32)
        # T2_fault,res_f = _get_T2_online(model,f_X,f_Y)
        # T_f_normal, _ = _get_T2_online(model, input[normal_index][trainX.shape[0]:], output[normal_index][trainX.shape[0]:])
        # f_X, f_Y,f_L = torch.sqrt(torch.sum(torch.square(f_X),1)),torch.sqrt(torch.sum(torch.square(f_Y),1)),torch.sqrt(torch.sum(torch.square(f_L),1))#,torch.sqrt(torch.sum(torch.sum(torch.square(f_J),2),1))
        # res = torch.sqrt(torch.sum(torch.square(torch.tensor(res,dtype=torch.float32)),1))
        # res_f = torch.sqrt(torch.sum(torch.square(res_f),1))

        # dirname = 't_sqk_'+m_name+'/'+m_name
        plt.rcParams['font.sans-serif'] = ['Times New Roman']
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  #
        # fig1 = plt.figure(figsize=(22, 9))
        # plt.scatter(f_X, f_Y, c='r',marker='+', s=20)
        # plt.scatter(n_X, n_Y,c='none',marker='o',edgecolors='b',s=5)
        # plt.scatter([], [], c='r',marker='+',s=100,label='故障样本')
        # plt.scatter([], [],  c='none',marker='o',edgecolors='b',s=100,label='无故障样本')
        # plt.legend(fontsize = 60, frameon = False,loc='lower right')#bbox_to_anchor=(0.85, 0.85)
        # plt.xlabel('$\Vert z \Vert_2$',fontproperties='Times New Roman',fontsize=60,style='italic')
        # plt.ylabel(r'$\Vert \hat z \Vert_2$',fontproperties='Times New Roman',fontsize=60,style='italic')
        # plt.yticks(fontproperties='Times New Roman', size=60)#,weight='bold'
        # plt.xticks(fontproperties='Times New Roman', size=60)
        #
        # plt.tight_layout()
        # # plt.title('DAEa_'+str(number),fontsize = 20)
        # plt.text(plt.xlim()[0],plt.ylim()[1] - 0.1*(plt.ylim()[1]-plt.ylim()[0]),'Model:'+m_name+' AMDR:'+str(model._perf[0][1])+'act: '+str(p['acts'][number-1][1:]),fontsize=20)
        # # plt.savefig(r"../Result/"+dirname+'_zhat_a'+str(number)+'.pdf', dpi=300, format="pdf")
        # plt.savefig(r"../Result/"+dirname+'_zhat_a'+str(number)+'.png', dpi=300, format="png")
        # # plt.show()
        #
        # fig2 = plt.figure(figsize=(15, 9))
        # # plt.scatter(f_X, f_J, c='r',s=3)
        # plt.scatter(n_X[:(len(n_J2))], n_J2, c='b',s=3)
        # # plt.scatter([], [], c='r',s=70,label='faulty')
        # plt.scatter([], [], c='b',marker='o',s=100,label='无故障样本')
        # plt.legend(fontsize = 45, frameon = False,loc='lower right')#bbox_to_anchor=(0.85, 0.85)
        # plt.xlabel('$\Vert z \Vert_2$',fontproperties='Times New Roman',fontsize=45,style='italic')
        # plt.ylabel(r'$\Vert \nabla _{uv}  \Vert_2$',fontproperties='Times New Roman',fontsize=45,style='italic')
        # plt.yticks(fontproperties='Times New Roman', size=45)#,weight='bold'
        # plt.xticks(fontproperties='Times New Roman', size=45)
        # # plt.ticklabel_format(style='plain')
        # plt.tight_layout()
        # plt.text(plt.xlim()[0],plt.ylim()[1] - 0.1*(plt.ylim()[1]-plt.ylim()[0]),'Model:'+m_name+' AMDR:'+str(model._perf[0][1])+'act: '+str(p['acts'][number-1][1:]),fontsize=20)
        # # plt.title('DAEa_'+str(number),fontsize = 20)
        # # plt.savefig(r"../Result/"+dirname+'_jacob_a'+str(number)+'.pdf', dpi=300, format="pdf")
        # plt.savefig(r"../Result/"+dirname+'_uv_jacob2_a'+str(number)+'.png', dpi=300, format="png")
        # # plt.show()
        #
        # fig12 = plt.figure(figsize=(15, 9))
        # # plt.scatter(f_X, f_J, c='r',s=3)
        # plt.scatter(n_X[[i for i in range(len(n_J1))]], n_J1, c='b',s=3)
        # # plt.scatter([], [], c='r',s=70,label='faulty')
        # plt.scatter([], [], c='b',marker='o',s=100,label='无故障样本')
        # plt.legend(fontsize = 45, frameon = False,loc='lower right')#bbox_to_anchor=(0.85, 0.85)
        # plt.xlabel('$\Vert z \Vert_2$',fontproperties='Times New Roman',fontsize=45,style='italic')
        # plt.ylabel(r'$\Vert \nabla _{uv}  \Vert_1$',fontproperties='Times New Roman',fontsize=45,style='italic')
        # plt.yticks(fontproperties='Times New Roman', size=45)#,weight='bold'
        # plt.xticks(fontproperties='Times New Roman', size=45)
        # # plt.ticklabel_format(style='plain')
        # plt.tight_layout()
        # plt.text(plt.xlim()[0],plt.ylim()[1] - 0.1*(plt.ylim()[1]-plt.ylim()[0]),'Model:'+m_name+' AMDR:'+str(model._perf[0][1])+'act: '+str(p['acts'][number-1][1:]),fontsize=20)
        # # plt.title('DAEa_'+str(number),fontsize = 20)
        # # plt.savefig(r"../Result/"+dirname+'_jacob_1_a'+str(number)+'.pdf', dpi=300, format="pdf")
        # plt.savefig(r"../Result/"+dirname+'_uv_jacob1_a'+str(number)+'.png', dpi=300, format="png")
        # plt.show()
        #
        # # z->T2
        normal = np.sqrt(np.sum(input[:trainX.shape[0]] **2,1))
        L = np.argmax(testY, 1)
        normal_loc, fault_loc = np.where(L == 0)[0], np.where(L != 0)[0]
        fault_T2 = model._TS[0]
        min_y = np.min(T2)

        fault = np.sqrt(np.sum(input[trainX.shape[0]:] ** 2, 1))
        fault_n, fault_f = fault[normal_loc], fault[fault_loc]

        T2_fn, T2_ff = fault_T2[normal_loc], fault_T2[fault_loc]

        min_y = min(min_y, np.min(T2_fn)) * 0.8

        min_x, max_x = np.min(normal) * 0.9, np.max(normal) *1.1
        loc = np.where(fault_f < max_x)
        max_y = np.max(T2_ff[loc]) * 1.2

        fig3 = plt.figure(figsize=(15, 9))

        # plt.scatter(f_X, f_Y, c='r',s=3)
        plt.scatter(normal, T2, c='none',marker='o',edgecolors='b',s=5)
        # plt.scatter(fault_n, T2_fn, c='none',marker='o',edgecolors='deepskyblue',s=5)
        plt.scatter(fault_f, T2_ff, c='r', marker='+', s=10, alpha=0.5)
        # plt.scatter(n_X[trainX.shape[0]:], T_f_normal, c='g',marker='+', s=10,alpha=0.5)
        # plt.scatter(n_X[:951], T2[:951],  c='none',marker='o',edgecolors='m',s=5)
        plt.scatter([], [], c='none',marker='o',edgecolors='b',s=100,label='正常样本')
        # plt.scatter([], [], c='none', marker='o', edgecolors='deepskyblue', s=100, label='测试集正常样本')
        plt.scatter([], [], c='r', marker='+', s=100, label='测试集故障样本')
        plt.plot([min_x, max_x], [thrd,thrd], '--', c='gray',label='阈值')
        plt.legend(fontsize = 25, frameon = False )#bbox_to_anchor=(0.85, 0.85)
        plt.xlabel('$\Vert z \Vert_2$',fontproperties='Times New Roman',fontsize=35,style='italic')
        plt.ylabel(r'$T^2$',fontproperties='Times New Roman',fontsize=35,style='italic')
        plt.yticks(fontproperties='Times New Roman', size=35)#,weight='bold'
        plt.xticks(fontproperties='Times New Roman', size=35)
        plt.yscale('log')
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)

        plt.tight_layout()
        # plt.title('DAEa_'+str(number),fontsize = 20)
        dirname = 'sqk/' + m_name
        plt.savefig(r"../Result/" + dirname + '_zT2_a' + str(number) + '.pdf', dpi=300, format="pdf")

        plt.text(plt.xlim()[0],plt.ylim()[1] - 0.2*(plt.ylim()[1]-plt.ylim()[0]),'Model:'+m_name+' AMDR:'+str(model._perf[0][1])+'act: '+str(p['acts'][number-1][1:]),fontsize=20)


        plt.savefig(r"../Result/"+dirname+'sssd'+str(dataset_id)+'_zT2_a'+str(number)+'.png', dpi=300, format="png")
        plt.show()
        plt.close()
        #
        # #r-mean(r)->T2
        # line_limit= torch.max(res)+1
        # fig4 = plt.figure(figsize=(15, 9))
        # # plt.scatter(f_X, f_Y, c='r',s=3)
        # plt.scatter(res, T2, c='none',marker='o',edgecolors='b',s=20,linewidths=0.5)
        # plt.scatter(res_f, T2_fault, c='r',marker='+', s=40,alpha=0.5,linewidths=0.5)
        # plt.scatter([], [], c='r',marker='+',s=100,label='故障样本')
        # plt.scatter([], [], c='none',marker='o',edgecolors='b',s=100,label='无故障样本')
        # plt.plot([0,line_limit], [thrd,thrd], '--', c='gray',label='阈值')
        # plt.legend(fontsize = 35, frameon = False,loc='lower right')#bbox_to_anchor=(0.85, 0.85)
        # plt.xlabel('$\Vert r \Vert_2$',fontproperties='Times New Roman',fontsize=45,style='italic')
        # plt.ylabel(r'$T^2$',fontproperties='Times New Roman',fontsize=45,style='italic')
        # plt.yticks(fontproperties='Times New Roman', size=45)#,weight='bold'
        # plt.xticks(fontproperties='Times New Roman', size=45)
        # plt.yscale('log')
        # plt.xlim(0, line_limit)
        # plt.tight_layout()
        # # plt.title('DAEa_'+str(number),fontsize = 20)
        # plt.text(plt.xlim()[0],plt.ylim()[1] - 0.2*(plt.ylim()[1]-plt.ylim()[0]),'Model:'+m_name+' AMDR:'+str(model._perf[0][1])+'act: '+str(p['acts'][number-1][1:]),fontsize=20)
        # # plt.savefig(r"../Result/"+dirname+'_rT2_a'+str(number)+'.pdf', dpi=300, format="pdf")
        # plt.savefig(r"../Result/"+dirname+'_rT2_a'+str(number)+'.png', dpi=300, format="png")
        # # plt.show()
        #
        # fig5 = plt.figure(figsize=(15, 9))
        # plt.scatter(f_X, res_f, c='r',marker='+', s=20,linewidths=1)
        # plt.scatter(n_X[:len(res)], res,c='none',marker='o',edgecolors='b',s=5,linewidths=1)
        # plt.scatter([], [], c='r',marker='+',s=100,label='故障样本')
        # plt.scatter([], [],  c='none',marker='o',edgecolors='b',s=100,label='无故障样本')
        # plt.legend(fontsize = 60, frameon = False,loc='lower right')#bbox_to_anchor=(0.85, 0.85)
        # plt.xlabel('$\Vert z \Vert_2$',fontproperties='Times New Roman',fontsize=60,style='italic')
        # plt.ylabel(r'$\Vert r \Vert_2$',fontproperties='Times New Roman',fontsize=60,style='italic')
        # plt.yticks(fontproperties='Times New Roman', size=60)#,weight='bold'
        # plt.xticks(fontproperties='Times New Roman', size=60)
        # plt.ylim(0, 5)
        # plt.xlim(0, 10)
        # plt.tight_layout()
        # # plt.title('DAEa_'+str(number),fontsize = 20)
        # plt.text(plt.xlim()[0],plt.ylim()[1] - 0.1*(plt.ylim()[1]-plt.ylim()[0]),'Model:'+m_name+' AMDR:'+str(model._perf[0][1])+'act: '+str(p['acts'][number-1][1:]),fontsize=20)
        # # plt.savefig(r"../Result/"+dirname+'_z2r_a'+str(number)+'.pdf', dpi=300, format="pdf")
        # plt.savefig(r"../Result/"+dirname+'_z2r_lim_a'+str(number)+'.png', dpi=300, format="png")

        # fig6 = plt.figure(figsize=(15, 9))
        # plt.scatter(f_X, f_L, c='r',marker='+', s=20,linewidths=1)
        # plt.scatter(n_X[:trainX.shape[0]], n_L[:trainX.shape[0]],c='none',marker='o',edgecolors='b',s=5,linewidths=1)
        # plt.scatter([], [], c='r',marker='+',s=100,label='故障样本')
        # plt.scatter([], [],  c='none',marker='o',edgecolors='b',s=100,label='无故障样本')
        # plt.legend(fontsize = 35, frameon = False,loc='lower right')#bbox_to_anchor=(0.85, 0.85)
        # plt.xlabel('$\Vert z \Vert_2$',fontproperties='Times New Roman',fontsize=60,style='italic')
        # plt.ylabel(r'$\Vert \xi  \Vert_2$',fontproperties='Times New Roman',fontsize=60,style='italic')
        # plt.yticks(fontproperties='Times New Roman', size=60)#,weight='bold'
        # plt.xticks(fontproperties='Times New Roman', size=60)
        # # plt.ylim(0, 5)
        # # plt.xlim(0, 10)
        # plt.tight_layout()
        # # plt.title('DAEa_'+str(number),fontsize = 20)
        # plt.text(plt.xlim()[0],plt.ylim()[1] - 0.1*(plt.ylim()[1]-plt.ylim()[0]),'Model:'+m_name+' AMDR:'+str(model._perf[0][1])+'act: '+str(p['acts'][number-1][1:]),fontsize=20)
        # # plt.savefig(r"../Result/"+dirname+'_z2r_a'+str(number)+'.pdf', dpi=300, format="pdf")
        # plt.savefig(r"../Result/"+dirname+'_z2latent_a'+str(number)+'.png', dpi=300, format="png")

        # fig7 = plt.figure(figsize=(15, 9))
        # # plt.scatter(f_X, f_L, c='r',marker='+', s=20,linewidths=1)
        # plt.scatter(n_X[:trainX.shape[0]], n_L[:trainX.shape[0]],c='none',marker='o',edgecolors='b',s=5,linewidths=1)
        # # plt.scatter([], [], c='r',marker='+',s=100,label='故障样本')
        # plt.scatter([], [],  c='none',marker='o',edgecolors='b',s=100,label='无故障样本')
        # plt.legend(fontsize = 35, frameon = False,loc='lower right')#bbox_to_anchor=(0.85, 0.85)
        # plt.xlabel('$\Vert z \Vert_2$',fontproperties='Times New Roman',fontsize=60,style='italic')
        # plt.ylabel(r'$\Vert \xi  \Vert_2$',fontproperties='Times New Roman',fontsize=60,style='italic')
        # plt.yticks(fontproperties='Times New Roman', size=60)#,weight='bold'
        # plt.xticks(fontproperties='Times New Roman', size=60)
        # # plt.ylim(0, 5)
        # # plt.xlim(0, 10)
        # plt.tight_layout()
        # # plt.title('DAEa_'+str(number),fontsize = 20)
        # plt.text(plt.xlim()[0],plt.ylim()[1] - 0.1*(plt.ylim()[1]-plt.ylim()[0]),'Model:'+m_name+' AMDR:'+str(model._perf[0][1])+'act: '+str(p['acts'][number-1][1:]),fontsize=20)
        # # plt.savefig(r"../Result/"+dirname+'_z2r_a'+str(number)+'.pdf', dpi=300, format="pdf")
        # plt.savefig(r"../Result/"+dirname+'_n_z2latent_a'+str(number)+'.png', dpi=300, format="png")

        # fig8 = plt.figure(figsize=(15, 9))
        # # plt.scatter(f_X, res_f, c='r',marker='+', s=20,linewidths=1)
        # plt.scatter(n_X[:trainX.shape[0]], h1[:trainX.shape[0]],c='none',marker='o',edgecolors='b',s=5,linewidths=1)
        # # plt.scatter([], [], c='r',marker='+',s=100,label='故障样本')
        # plt.scatter([], [],  c='none',marker='o',edgecolors='b',s=100,label='无故障样本')
        # plt.legend(fontsize = 60, frameon = False,loc='lower right')#bbox_to_anchor=(0.85, 0.85)
        # plt.xlabel('$\Vert z \Vert_2$',fontproperties='Times New Roman',fontsize=60,style='italic')
        # plt.ylabel(r'$\Vert hidden1 \Vert_2$',fontproperties='Times New Roman',fontsize=60,style='italic')
        # plt.yticks(fontproperties='Times New Roman', size=60)#,weight='bold'
        # plt.xticks(fontproperties='Times New Roman', size=60)
        # # plt.ylim(0, 5)
        # # plt.xlim(0, 10)
        # plt.tight_layout()
        # # plt.title('DAEa_'+str(number),fontsize = 20)
        # plt.text(plt.xlim()[0],plt.ylim()[1] - 0.1*(plt.ylim()[1]-plt.ylim()[0]),'Model:'+m_name+' AMDR:'+str(model._perf[0][1])+'act: '+str(p['acts'][number-1][1:]),fontsize=20)
        # # plt.savefig(r"../Result/"+dirname+'_z2r_a'+str(number)+'.pdf', dpi=300, format="pdf")
        # plt.savefig(r"../Result/"+dirname+'_z2hidden_1_a'+str(number)+'.png', dpi=300, format="png")

        #fault10的正常和异常样本
        # fig8 = plt.figure(figsize=(15, 9))
        # # plt.scatter(f_X, res_f, c='r',marker='+', s=20,linewidths=1)
        # normal_feature = input[normal_index][:, 7]
        # fault_feature = input[fault_index][:, 7]
        # plt.scatter(list(range(len(normal_feature))),normal_feature,c='none',marker='o',edgecolors='b',s=5,linewidths=2)
        # plt.scatter(list(range(len(normal_feature),len(normal_feature)+len(fault_feature))), fault_feature, c='r',marker='+', s=20,linewidths=2,alpha=0.6)
        # plt.scatter([], [], c='r',marker='+',s=100,label='故障样本')
        # plt.scatter([], [],  c='none',marker='o',edgecolors='b',s=100,label='无故障样本')
        # plt.axvline(x=len(normal_feature)+len(fault_feature)-1000, ymin=plt.ylim()[0],ymax = plt.ylim()[1],color = 'k',linestyle = 'dashed')
        # plt.legend(fontsize = 20, frameon = False,loc='lower right')#bbox_to_anchor=(0.85, 0.85)
        # plt.xlabel('$\Vert z \Vert_2$',fontproperties='Times New Roman',fontsize=20,style='italic')
        # plt.ylabel(r'$\Vert hidden1 \Vert_2$',fontproperties='Times New Roman',fontsize=20,style='italic')
        # plt.yticks(fontproperties='Times New Roman', size=20)#,weight='bold'
        # plt.xticks(fontproperties='Times New Roman', size=20)
        # # plt.ylim(0, 5)
        # # plt.xlim(0, 10)
        # plt.tight_layout()
        # # plt.title('DAEa_'+str(number),fontsize = 20)
        # # plt.text(plt.xlim()[0],plt.ylim()[1] - 0.1*(plt.ylim()[1]-plt.ylim()[0]),'Model:'+m_name+' AMDR:'+str(model._perf[0][1])+'act: '+str(p['acts'][number-1][1:]),fontsize=20)
        # # # plt.savefig(r"../Result/"+dirname+'_z2r_a'+str(number)+'.pdf', dpi=300, format="pdf")
        # plt.savefig(r"../Result/"+dirname+'fault10的正常和异常样本'+str(number)+'.png', dpi=300, format="png")
        # plt.close()
#
# import matplotlib.pyplot as plt
# import numpy as np
# def sigmoid(x):
#     y = 1.0 / (1.0 + np.exp(-x))
#     return y
# def gauss(x):
#     y = 1 -  np.exp(-x**2)
#     return y
#
# fig3 = plt.figure(figsize=(15, 9))
# plot_x = np.linspace(-3, 3, 1000) # 绘制范围[-10,10]
# # plot_y = sigmoid(plot_x)
# # plot_y = np.tanh(plot_x)
# plot_y = gauss(plot_x)
# plt.plot(plot_x, plot_y,'black', linewidth=1)
# # plt.plot(plot_x, plot_x,'b', linewidth=1)
# # plt.xlim(-10,10)
# # plt.ylim(0,1)
# # plt.xlim(-10,10)
# # plt.ylim(-1.5,1.5)
# # plt.text(plt.xlim()[0],plt.ylim()[1],'sigmoid/y=x 0.65',fontsize=30)
# # plt.savefig(r"../Result/"+'sigmoid'+'.png', dpi=400, format="png")
# # plt.text(plt.xlim()[0],plt.ylim()[1],'tanh/y=x',fontsize=30)
# # plt.savefig(r"../Result/"+'tanh'+'.png', dpi=400, format="png")
# plt.text(plt.xlim()[0],plt.ylim()[1],'gauss/y=x',fontsize=30)
# plt.savefig(r"../Result/"+'1gauss'+'.png', dpi=400, format="png")
# plt.show()
