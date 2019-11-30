
## Train
1. for first run, you can call gen_pattern() to generate DMD pattern.
There is an already generated pattern "p1127.txt". Use it directly.   

2. unzip "test_images.zip" from MNIST dataset.

3. use the following command to train a CNN on Mnist
    python main.py --datapath=data/test_images
    　　　　　　　　--batch_size 256
    　　　　　　　　--epochs 10
    　　　　　　　　--no_cuda True
    

## Test
Run `test.py`  
make sure the following 2 lines are consistent with the training code before testing:
1. model = SPNet('p1127.txt', 128)
2. model.load_state_dict(torch.load('output/params_14.pth'))
