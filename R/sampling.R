library(tidyverse)

user = read.csv("./原数据-20210301-20210328/users.txt", sep = '\t', header=T, encoding = 'UTF-8')

poi = read.csv("./原数据-20210301-20210328/pois.txt", sep = '\t', header=T, encoding = 'UTF-8')

order_poi = read.csv("./原数据-20210301-20210328/orders_poi_session.txt", 
                       sep = '\t', header=T, encoding = 'UTF-8')

order_train = read.csv("./原数据-20210301-20210328/orders_train.txt",
                       sep = '\t', header=T, encoding = 'UTF-8')

order_test = read.csv("./原数据-20210301-20210328/orders_test_poi.txt",
                      sep = '\t', header=T, encoding = 'UTF-8')

order_test_label = read.csv("./原数据-label-20210301-20210328/orders_poi_test_label.txt",
                            sep = '\t', header=T, encoding = 'UTF-8')

user_idx = sample(user$user_id, size = 20000)

user2 = user %>% 
  filter(user_id %in% user_idx)

order_train2 = order_train %>% 
  filter(user_id %in% user_idx)

order_test2 = order_test %>% 
  filter(user_id %in% user_idx)

order_test_label2 = order_test_label %>% 
  filter(user_id %in% user_idx)

order_poi2 = order_poi %>% 
  filter(wm_order_id %in% unique(order_train2$wm_order_id))

poi2 = poi %>% 
  filter(wm_poi_id %in% unique(order_train2$wm_poi_id))

write.csv(user2, file = "./dat/user_sampling.csv", fileEncoding = "UTF-8", row.names = F)
write.csv(order_train2, file = "./dat/order_train_sampling.csv", fileEncoding = "UTF-8", row.names = F)
write.csv(order_test2, file = "./dat/order_test_sampling.csv", fileEncoding = "UTF-8", row.names = F)
write.csv(order_poi2, file = "./dat/orders_poi_session_sampling.csv", fileEncoding = "UTF-8", row.names = F)
write.csv(poi2, file = "./dat/poi_sampling.csv", fileEncoding = "UTF-8", row.names = F)
