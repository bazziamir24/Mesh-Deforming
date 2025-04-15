import tensorflow as tf 
import sonnet as snt

class MyLinear(snt.Module):
    def __init__(self, output_size, name = None):
        super(MyLinear, self).__init__(name = name)
        self.output_size = output_size
        
    @snt.once
    def _initialize(self, x):
        initial_w = tf.random.normal([x.shape[1], self.output_size])
        self.w = tf.Variable(initial_w, name = "w")
        self.b = tf.Variable(tf.zeros([self.output_size]), name = "b")
        
    def __call__(self,x):
        self._initialize(x)
        return tf.matmul(x, self.w) + self.b
    
if __name__ == "__main__":
    batch_size = 8
    input_size = 728
    mod = MyLinear(32)
    output = mod(tf.ones([batch_size, input_size]))
    print(output.shape)
    print(mod)
    print(mod.variables)
    