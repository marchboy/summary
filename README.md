

## Date: Week6-homework

**遇到问题**

Win X64 bit：

```
RuntimeError: No CUDA GPUs are available
```

![](photo/week6_1.png)

**解决方案**

在文件代码`train_extractive.py`中第217-219行中：

```python
# if device_id >= 0:
#     torch.cuda.set_device(device_id)
#     torch.cuda.manual_seed(args.seed)
```

注释掉这段，这是多卡训练时，控制用哪几张GPU的指令，如果你是单卡，直接删掉就可以了。



**Result**

![](photo/week6_2.png)

![](photo/week6_3.png)

## Date: Week5-homework

![](photo/week5.png)


## Date: Week4-homework

图片较大，加载可能比较慢。

![](photo/微信截图_20210524011648.png)

![](photo/微信截图_20210524011715.png)

![](photo/微信截图_20210524011749.png)

## windows环境问题小结

一、缺少python-Levenshtein包，windows无法安装的问题。

解决方案：https://www.lfd.uci.edu/~gohlke/pythonlibs/里面查找python-Levenshtein.(版本号).whl，然后可以pip intall了

二、没有安装gensim==3.8.3的包，而是默认安装了最新的4.0.1版本的包。

需要将Word2Vec方法的参数size改为vector_size，iter改为epochs

三、找不到src模块

以绝对路径或者相对路径导入文件的方法
```
import sys
sys.path.append("D:/Projects/TextSummary/week2_项目导论中与中文词向量实践/summary/")
```


## 使用步骤
先将AutoMaster_TrainSet 和 AutoMaster_TestSet 拷贝到data 路径下 再使用 .


代码结构

+ result 结果保存路径
    ....    
+ seq2seq_tf2 模型结构
    ....
+ utils 工具包
    + config  配置文件
    + data_loader 数据处理模块
    + multi_proc_utils 多进程数据处理
+ data  数据集
    + AutoMaster_TrainSet 拷贝数据集到该路径
    + AutoMaster_TestSet  拷贝数据集到该路径
    ....
    
    

训练步骤:
1. 拷贝数据集到data路径下
2. 运行utils\data_loader.py可以一键完成 预处理数据 构建数据集

##  seq2seq_tf2 模块
* 训练模型 运行seq2seq_tf2\train.py脚本,进入 summary 目录,运行如下命令:
    ```bash
    $ python -m src.seq2seq_tf2.train
    ```


预测步骤:
1. greedy decode 和 beam search 的代码都在 predict_helper.py 中，greedy 使用的是验证集损失最小的
    ```json
   {
        "rouge-1": { 
           "f": 0.31761580523281824,
           "p": 0.35095753378433,  
           "r": 0.3439340546952935
       },
       "rouge-2": {
           "f": 0.13679872398179568,
           "p": 0.14990364277693116,
           "r": 0.1455355455469621
       },
       "rouge-l": {    
           "f": 0.3193850357115565,
           "p": 0.3712312939226222,
           "r": 0.31313195314734876
       }
   }
   ```
2. 运行 predict.py 调用 greedy decode 或者 beam search 进行预测，beam search 使用的是最后一个ckpt，所以可能差在这

    ```json
    {
      "rouge-1": {
        "f": 0.2915635508381314,
        "p": 0.3804116780031509,
        "r": 0.2719187833527539
      },
      "rouge-2": {
        "f": 0.12879757845176937,
        "p": 0.16785280802905464,
        "r": 0.11995412911919483
      },
      "rouge-l": {
        "f": 0.2901830986437813,
        "p": 0.3689693518270286,
        "r": 0.26824690814678
      }
    }
    ```
## pgn_tf2 模块
1. 使用了 pointer ，未使用 coverage 机制，
   - greedy 解码
       ```json
        {
          "rouge-1": {
            "f": 0.2829960170084011,
            "p": 0.38460960847631137,
            "r": 0.2845213443094316
          },
          "rouge-2": {
            "f": 0.12414605543620791,
            "p": 0.1676379768113762,
            "r": 0.1269054024767184
          },
          "rouge-l": {
            "f": 0.29623175018160924,
            "p": 0.42430977114926877,
            "r": 0.26538800605593477
          }
        }
       ```
   - beam search
       ```json
        {
          "rouge-1": {
               "f": 0.2651999864228139,
               "p": 0.4065036840056818,
               "r": 0.2469963017215841
          },
          "rouge-2": {
               "f": 0.12424685470610693,
               "p": 0.195351585337393,
               "r": 0.11686033974071998
          },
          "rouge-l": {
               "f": 0.2822344430219466,
               "p": 0.4295556625392504,
               "r": 0.242722479908954
          }
        }
       ```
2. pointer + coverage
   - greedy 解码
       ```json
         {
           "rouge-1": {
             "f": 0.29464152746519856,
             "p": 0.37752092544374083,
             "r": 0.30440518835181823
           },
           "rouge-2": {
             "f": 0.1335177727081173,
             "p": 0.16973912800870636,
             "r": 0.13972597888187344
           },
           "rouge-l": {
             "f": 0.29598713909552105,
             "p": 0.38594656409806355,
             "r": 0.2820867442173792
           }
         }
       ```
## pgn_transformer_tf2 模块
1. max_enc_len = 200,max_dec_len = 40
   ```json
    {
      "rouge-1": {
        "f": 0.12487834161314122,
        "p": 0.10019451938562356,
        "r": 0.24147098362099628
      },
      "rouge-2": {
        "f": 0.03309384504232345,
        "p": 0.026344187912833376,
        "r": 0.07200730112164937
      },
      "rouge-l": {
        "f": 0.24297878453744742,
        "p": 0.40690526450124065,
        "r": 0.20128083026164661
      }
    }
   ```
2. max_enc_len=400, max_dec_len = 100

   ```json
    {
      "rouge-1": {
        "f": 0.08574490221087681,
        "p": 0.1318115568560623,
        "r": 0.1788133913191769
      },
      "rouge-2": {
        "f": 0.01576017532651374,
        "p": 0.030049816495901254,
        "r": 0.032510132165779596
      },
      "rouge-l": {
        "f": 0.19713579322392735,
        "p": 0.40649808194131554,
        "r": 0.14962353101026613
      }
    }
   ```

