library(tidyverse)

user = read.csv("./dat/user_sampling.csv",  header=T, encoding = 'UTF-8')

poi = read.csv("./dat/poi_sampling.csv",  header=T, encoding = 'UTF-8')

poi$primary_second_tag_name = as.numeric(poi$primary_second_tag_name)
poi$primary_first_tag_name = str_replace_all(poi$primary_first_tag_name, " ", "")
poi$primary_third_tag_name = str_replace_all(poi$primary_third_tag_name, "\t", "")

as.character(as.numeric(poi$primary_third_tag_name)) %>% table()

order_poi = read.csv("./dat/orders_poi_session_sampling.csv", 
                      header=T, encoding = 'UTF-8')

longer = function(r){
  r = as.vector(r)
  t = unlist(str_split(r[2], "#"))
  t = unique(t)
  df = data.frame(wm_order_id = r[1],
             click_poi = t, 
             dt = r[3])
  df = apply(df, 2, as.numeric)
  return(df)
}
longer(order_poi[1,])

clicks = apply(order_poi, 1, longer) %>% 
  do.call(rbind,.)

clicks = na.omit(clicks)
clicks = as.data.frame(clicks)

order_train = read.csv("./dat/order_train_sampling.csv",  header=T, encoding = 'UTF-8')

clicks = clicks %>% 
  left_join(select(order_train, wm_order_id, user_id)) %>% 
  select(user_id, wm_poi_id = click_poi, dt) %>% 
  mutate(type = "click")

#clicks = cbind(clicks, type = "click")

write.csv(clicks, "./dat/clicks.csv", fileEncoding = "UTF-8", row.names = F)
