# Pretest for asu research position
### This is split up into two sections, part 1 is matlab and part 2 is a neural network

### Matlab section
The idea was, given a k-sparse vector (n,1), generate a random sensing matrix A (m,n), and multiply 
them to get Y=AX (m,1).

Once you have Y, use gradient descent to find the original X, using only Y and the sensing matrix A.


### Neural network
The first part was to compress the MNIST numbers dataset, using a sensing matrix  
from 28x28 pixel images to 7x7 pixels.  
Then, using a simple neural net, recreate the original picture, at 28x28 pixels  

The images are related 
before- image before compression and reconstruction  
![alt text](https://github.com/nick-moran/compressive_sensing_pretest/blob/master/before.png?raw=true)  
inter- image after compression, but before reconstruction  
![alt text](https://github.com/nick-moran/compressive_sensing_pretest/blob/master/inter.png?raw=true)  
after- image after compression and reconstruction  
![alt text](https://github.com/nick-moran/compressive_sensing_pretest/blob/master/after.png?raw=true)  

