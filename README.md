# Benchmarking-ES-Atari
##Installation

Cần cài đặt các thư viện sau 

pytorch 

gym

##Training 

Chạy file ```examples/space_invaders/train_image.py``` nếu sử dụng đầu vào là image, ```examples/space_invaders/train_ram.py``` nếu sử dụng đầu vào là ram

Ví dụ:

```shell
python examples/space_invaders/train_image.py --weights_path model_space_invaders_image.p --cuda 
```

Các tham số về population_size, sigma, learning_rate thay đổi trong hàm ```EvolutionModule```





