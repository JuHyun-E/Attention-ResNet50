from train import *

# 학습이 끝나면 테스트 데이터(10000개)에 대한 정확도를 출력합니다.
test_accuracy = 0.0
for i in range(10):
    test_batch = next_batch(1000, x_test, y_test_one_hot.eval())
    test_accuracy = test_accuracy + accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1.0})
test_accuracy = test_accuracy / 10;
print("테스트 데이터 정확도: %f" % test_accuracy)