**Samples**
 - This is the len(dataX), or the amount of data points you have.

**Time steps**
 - This is equivalent to the amount of time steps you run your RNN. 
If you want your network to have memory of 60 characters, this number should be 60.

**Features**
- this is the amount of features in every time step. If you are processing pictures, this is the amount of pixels.
In this case you seem to have 1 feature per time step

***Spliting***
Since we are feeding line by line into the NN, it might be best to just use the '/s' space as a separator. F.ex look at this line: 
``` 
mov DWORD PTR -4[rbp], edi
```