import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

# 加载预训练的人脸生成模型
generate = hub.Module("https://tfhub.dev/google/progan-128/1")

# 生成人脸图像
latent_vector = tf.random.normal([20, 512])  # 随机生成一个潜在向量
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    generated_image = sess.run(generate(latent_vector))

# 显示生成的人脸图像
plt.imshow((generated_image[0] * 255).astype("uint8"))
plt.axis('off')
plt.show()
