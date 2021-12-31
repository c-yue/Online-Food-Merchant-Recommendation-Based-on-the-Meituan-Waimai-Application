#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os


# In[260]:


# os.chdir('C:/Users/Axe-Yue-CHEN/Desktop/Online-Food-Merchant-Recommendation-Based-on-the-Meituan-Waimai-Application')


# # 1 Data Preparing

# ## 1.1 Read Data

# In[261]:


# read user


# In[110]:


path = './原数据-20210301-20210328/users.txt'
users = pd.read_csv(path, sep = "\t")


# In[111]:


users['avg_pay_amt'].apply(lambda x: str(x).split(','))


# In[ ]:





# In[112]:


# read pois
path = './原数据-20210301-20210328/pois.txt'
pois = pd.read_csv(path, sep = "\t")


# In[113]:


# read orders


# In[114]:


path = './原数据-20210301-20210328/orders_train.txt'
orders_train = pd.read_csv(path, sep = "\t")
# path = './dat/order_train_sampling.csv'
# orders_train = pd.read_csv(path)


# In[115]:


orders_train.head()
# orders_train


# In[116]:


orders_train


# In[117]:


min(orders_train['dt'])


# In[ ]:





# In[118]:


path = './原数据-label-20210301-20210328/orders_poi_test_label.txt'
orders_test = pd.read_csv(path, sep = "\t")


# In[119]:


orders_test.head()


# In[120]:


# test if there is user never seen 
# result: no user never seen before
sum(pd.merge(orders_test, users, on = 'user_id', how='left')['avg_pay_amt'].isnull())


# In[121]:


# test if there is poi never seen 
# result: no poi never seen before
sum(pd.merge(orders_test, pois, on = 'wm_poi_id', how='left')['wm_poi_name'].isnull())


# In[122]:


len(orders_test['user_id'].unique())


# In[ ]:





# ## 1.2 Cleaning Data

# In[123]:


# pois


# In[124]:


pois = pois[0:29070]


# In[ ]:





# ## 1.3 Exploring

# ### Most Bought POI

# In[125]:


order_sort_by_poi = orders_train.groupby('wm_poi_id').count().sort_values('user_id',ascending = False)


# In[288]:


order_sort_by_poi.head()


# In[ ]:





# ## 1.4 Data preparing

# In[130]:


# user without info
users_n_info = users['user_id'].to_frame()
# users_n_info = users_n_info [:10000]


# In[282]:


users.head()


# In[283]:


user_index = {user: user for user in users['user_id']}


# In[ ]:





# In[284]:


# order record
user_poi_buy = orders_train[['user_id','wm_poi_id']]
# user_poi_buy = user_poi_buy [:3000]


# In[285]:


user_poi_buy['if_buy'] = 1


# In[286]:


pois['wm_poi_id'] = pois['wm_poi_id'].astype('int')


# In[384]:


poi_index = {poi: poi for poi in pois['wm_poi_id']}


# In[ ]:





# # 2 Recall Model

# 2.1 Embedding: Get Candidate Set 1 by Embedding
# 
# 2.2 Click History: Get Candidate Set 2 by extracting click history
# 
# 2.3 Geo Relation: Get Candidate Set 3 by extracting include the user - merchant in the candidate set if the matches happened most in history. Here, we include the top 20

# ## 2.1 Embedding 

# **Embedding model needs a training set.** We are going to treat this as a supervised learning problem: given a pair (user, poi), we want the neural network to learn to predict whether this is a legitimate pair - present in the data - or not.

# In[137]:


# positive links between users and pois


# In[138]:


user_poi_buy.head()


# In[139]:


pos_link_list = user_poi_buy[['user_id','wm_poi_id']].values.tolist()
pos_pairs = [tuple(link) for link in pos_link_list]
pos_pairs_set = set(pos_pairs)


# In[140]:


len(pos_pairs_set)


# In[141]:


len(pos_link_list)


# In[187]:


user_poi_num = user_poi_buy.groupby(['user_id','wm_poi_id'],as_index=False).sum('if_buy')
user_poi_num_list = user_poi_num[['user_id','wm_poi_id','if_buy']].values.tolist()


# In[188]:


user_poi_num[user_poi_num['user_id'] == 1000]


# In[189]:


from collections import defaultdict
user_poi_num_dict = defaultdict(dict)
for i in user_poi_num_list:
    user_poi_num_dict[i[0]][i[1]] = i[2]


# In[ ]:





# In[144]:


from collections import defaultdict
pos_pairs_dict = defaultdict(dict)
for k,v in pos_pairs_set:
    dict2 = {}
    dict2[v] = 1
    pos_pairs_dict[k][v] = 1


# In[145]:


pos_pairs_dict[1]


# In[146]:


user_poi_buy[user_poi_buy['user_id']==1]['wm_poi_id'].unique()


# In[ ]:





# In[38]:


# negative links between users and pois


# In[39]:


# user_poi_0


# In[40]:


# neg_link_list = user_poi_0[['user_id','wm_poi_id']].values.tolist()
# neg_pairs = [tuple(link) for link in neg_link_list]
# neg_pairs_set = set(neg_pairs)


# In[41]:


# len(neg_pairs_set)


# To compute the embeddings, we are **not going to create a separate validation or testing set.** While this is a must for a normal supervised machine learning task, in this case, our primary objective is not to make the most accurate model, but to generate the best embeddings. The prediction task is just the method through which we train our network to make the embeddings. At the end of training, we are not going to be testing our model on new data, so we don't need to evaluate the performance. 

# The code below creates **a generator that yields batches of samples each time it is called**. Neural networks are trained incrementally - a batch at a time - which means that a generator is a useful function for returning examples on which to train. 

# In[147]:


import random
random.seed(100)
def generate_batch(pairs, n_positive = 50, negative_ratio = 1.0, classification = True):
    """Generate batches of samples for training"""
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))
    
    # Adjust label based on task
    if classification:
        neg_label = 0
    else:
        neg_label = -1
    
    # This creates a generator
    while True:
        # randomly choose positive examples
        for idx, (user_id, wm_poi_id) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (user_id, wm_poi_id, 1)

        # Increment idx by 1
        idx += 1
        
        # Add negative examples until reach batch size
        while idx < batch_size:
            
            # random selection
            random_user = random.randrange(len(users))
            random_poi = random.randrange(len(pois))
            
            # Check to make sure this is not a positive example
            if (random_user, random_poi) not in pos_pairs_set:
                
                # Add to batch and increment index
                batch[idx, :] = (random_user, random_poi, neg_label)
                idx += 1
                
        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {'user': batch[:, 0], 'poi': batch[:, 1]}, batch[:, 2]


# In[148]:


next(generate_batch(pos_pairs, n_positive = 2, negative_ratio = 3))


# **Neural Network Embedding Model**

# In[149]:


# pip install keras


# In[150]:


# pip install tensorflow


# In[151]:


import keras
from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model


# In[152]:


def user_embedding_model(embedding_size = 50, classification = False):
    """Model to embed users and pois using the functional API.
        Trained to discern if a link is present in the user poi pairs (which is orders)"""
    
    # Both inputs are 1-dimensional
    user = Input(name = 'user', shape = [1])
    poi = Input(name = 'poi', shape = [1])
    
    # Embedding the user (shape will be (None, 1, 50))
    user_embedding = Embedding(name = 'user_embedding',
                                input_dim = len(users['user_id']),
                                output_dim = embedding_size)(user)
    
    # Embedding the poi (shape will be (None, 1, 50))
    poi_embedding = Embedding(name = 'poi_embedding',
                                input_dim = len(pois['wm_poi_id']),
                                output_dim = embedding_size)(poi)
    
    # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
    merged = Dot(name = 'dot_product', normalize = True, axes = 2)([user_embedding, poi_embedding])
    
    # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape = [1])(merged)
    
    # If classifcation, add extra layer and loss function is binary cross entropy
    if classification:
        merged = Dense(1, activation = 'sigmoid')(merged)
        model = Model(inputs = [user, poi], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Otherwise loss function is mean squared error
    else:
        model = Model(inputs = [user, poi], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'mse')
    
    return model

# Instantiate model and show parameters
model = user_embedding_model()
model.summary()


# **Train Embedding Model**

# In[154]:


n_positive = 1024

gen = generate_batch(pos_pairs, n_positive, negative_ratio = 2)

# Train
h = model.fit(x = gen, epochs = 15,steps_per_epoch = len(pos_pairs) // n_positive,verbose = 2)


# In[ ]:


model.save('./models/first_attempt.h5')


# In[155]:


model = keras.models.load_model('./models/first_attempt.h5')


# **Extract Embeddings and Analyze**

# In[396]:


# Extract embeddings
user_layer = model.get_layer('user_embedding')
user_weights = user_layer.get_weights()[0]
user_weights.shape


# Each user is now represented as a 50-dimensional vector.
# 
# We need to normalize the embeddings so that the dot product between two embeddings becomes the cosine similarity.

# In[397]:


user_weights = user_weights / np.linalg.norm(user_weights, axis = 1).reshape((-1, 1))
user_weights[0][:10]
np.sum(np.square(user_weights[0]))


# Normalize just means divide each vector by the square root of the sum of squared components.

# **Finding Similar users**

# Function to Find Most Similar Entities

# The function below takes in either a user or a poi, a set of embeddings, and returns the n most similar items to the query. It does this by computing the dot product between the query and embeddings. Because we normalized the embeddings, the dot product represents the cosine similarity between two vectors. This is a measure of similarity that does not depend on the magnitude of the vector in contrast to the Euclidean distance.
# 
# Once we have the dot products, we can sort the results to find the closest entities in the embedding space. With cosine similarity, higher numbers indicate entities that are closer together, with -1 the furthest apart and +1 closest together.

# In[398]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 15

def find_similar(name, weights, index_name = 'user', n = 10, least = False, return_dist = False, plot = False):
    """Find n most similar items (or least) to name based on embeddings. Option to also plot the results"""
    
    # Select index and reverse index
    if index_name == 'user':
        index = user_index
    elif index_name == 'poi':
        index = poi_index
    
    # Check to make sure `name` is in index
    try:
        # Calculate dot product between book and all others
        dists = np.dot(weights, weights[index[name]])
    except KeyError:
        print(f'{name} Not Found.')
        return
    
    # Sort distance indexes from smallest to largest
    sorted_dists = np.argsort(dists)
    
    # Plot results if specified
    if plot:
        
        # Find furthest and closest items
        furthest = sorted_dists[:(n // 2)]
        closest = sorted_dists[-n-1: len(dists) - 1]
        items = [index[c] for c in furthest]
        items.extend(index[c] for c in closest)
        
        # Find furthest and closets distances
        distances = [dists[c] for c in furthest]
        distances.extend(dists[c] for c in closest)
        
        colors = ['r' for _ in range(n //2)]
        colors.extend('g' for _ in range(n))
        
        data = pd.DataFrame({'distance': distances}, index = items)
        
        # Horizontal bar chart
        data['distance'].plot.barh(color = colors, figsize = (10, 8),
                                    edgecolor = 'k', linewidth = 2)
        plt.xlabel('Cosine Similarity');
        plt.axvline(x = 0, color = 'k');
        
        # Formatting for italicized title
        name_str = f'{index_name.capitalize()}s Most and Least Similar to'
        for word in str(name).split():
            # Title uses latex for italize
            name_str += ' $\it{' + word + '}$'
        plt.title(name_str, x = 0.2, size = 28, y = 1.05)
        
        return None
    
    # If specified, find the least similar
    if least:
        # Take the first n from sorted distances
        closest = sorted_dists[:n]
        print(f'{index_name.capitalize()}s furthest from {name}.\n')
        
    # Otherwise find the most similar
    else:
        # Take the last n sorted distances
        closest = sorted_dists[-n:]
        
        # Need distances later on
        if return_dist:
            return dists, closest
        
        
        print(f'{index_name.capitalize()}s closest to {name}.\n')
        
    # Need distances later on
    if return_dist:
        return dists, closest
    
    
    # Print formatting
    max_width = 6
    
    # Print the most similar and distances
    for c in reversed(closest):
        print(f'{index_name.capitalize()}: {index[c]:{max_width + 2}} Similarity: {dists[c]:.{2}}')
        
    


# In[399]:


find_similar(20000, user_weights, index_name='user', n = 10, least = False)


# **Finding Similar pois**

# In[400]:


def extract_weights(name, model):
    """Extract weights from a neural network model"""
    
    # Extract weights
    weight_layer = model.get_layer(name)
    weights = weight_layer.get_weights()[0]
    
    # Normalize
    weights = weights / np.linalg.norm(weights, axis = 1).reshape((-1, 1))
    return weights


# In[401]:


poi_weights = extract_weights('poi_embedding', model)


# In[402]:


find_similar(20000, poi_weights, index_name='poi', n = 10, least = False)


# In[403]:


find_similar(20000, poi_weights, index_name='poi', n=5, plot=True)


# In[ ]:





# In[ ]:





# ## Classification Model

# For this model, the negative examples receive a label of 0 and the loss function is binary cross entropy. The procedure for the neural network to learn the embeddings is exactly the same, only it will be optimizing for a slightly different measure.

# In[164]:


model_class = user_embedding_model(50, classification = True)
n_positive = 1024
gen = generate_batch(pos_pairs_set, n_positive, negative_ratio=2, classification = True)


# In[80]:


# Train the model to learn embeddings
h = model_class.fit_generator(gen, epochs = 15, steps_per_epoch= len(pos_pairs_set) // n_positive, verbose = 0)


# In[390]:


model_class.save('./models/first_attempt_class.h5')


# In[404]:


model_class = keras.models.load_model('./models/first_attempt_class.h5')


# In[405]:


user_weights_class = extract_weights('user_embedding', model_class)
user_weights_class.shape


# In[406]:


find_similar(20000, user_weights_class, n = 5)


# Things are looking pretty good with this model as well.

# In[407]:


poi_weights_class = extract_weights('poi_embedding', model_class)


# In[408]:


find_similar(20000, poi_weights_class, index_name = 'poi', n = 5)


# In[ ]:





# Ouput poi weight and user weight:

# In[409]:


poi_weight_df = pd.DataFrame()
poi_weight_df['wm_poi_id'] = range(len(poi_weights_class))


# In[410]:


poi_weight_df = pd.merge(poi_weight_df, pd.DataFrame(poi_weights_class),left_index=True, right_index=True)


# In[411]:


poi_weight_df


# In[412]:


user_weight_df = pd.DataFrame()
user_weight_df['user_id'] = range(len(user_weights_class))


# In[413]:


user_weight_df = pd.merge(user_weight_df, pd.DataFrame(user_weights_class),left_index=True, right_index=True)


# In[414]:


user_weight_df


# In[415]:


poi_weight_df.to_csv("dat/poi_weight.csv", index=False)
user_weight_df.to_csv("dat/user_weight.csv", index=False)


# In[ ]:





# ## U2I (U2POI)

# In[596]:


def poi_recommend_4_user(user, user_weights, n_sim_user, n_top_poi):
    # user: the user to get recommend poi
    # user_weights: embedded user weight, this is to calculate user similarity
    # n_sim_user
    # n_top_poi: Top n pois to return
    # Gets recommendations for a person by using a weighted average of every other user's rankings
    
    # Tried for sevearal a times, for a curve and it turns out that bought pois of the most similar xxx users' perform well
    
    totals = {}
    sim_sum = {}
    
    rankings_list =[]
    
    dists = np.dot(user_weights, user_weights[user_index[user]])
    sim_user_index = dists.argsort()[::-1][0:n_sim_user] 
    
    for other_user in sim_user_index:
        user_sim = dists[other_user]
        pois_bought = user_poi_num_dict[other_user].keys()
        pois_bought_tt = sum(user_poi_num_dict[other_user].values())
        
        for poi in pois_bought:
            # Similrity * score
            totals.setdefault(poi,0)
            # make sure to extract 0 though there is no this value
            user_poi_num_dict[other_user].setdefault(poi,0)
            totals[poi] += user_poi_num_dict[other_user][poi] * user_sim
            # print(totals[poi])
            
            # sum of similarities
            sim_sum.setdefault(poi,0)
            sim_sum[poi]+= user_sim
            # print(sim_sum[poi])
        
        # print('loop done for other user:')
        # print(other_user)
    
    rankings = [(total/sim_sum[poi],poi) for poi,total in totals.items()]
    rankings.sort()
    rankings.reverse()
    recommend_list = [recommend_item for score,recommend_item in rankings]
    return recommend_list[0:n]


# In[597]:


import random
random.seed(20211221)
sample_users = random.sample(user_index.keys(), 10000)


# In[598]:


# pip install line_profiler


# In[600]:


sim_user_eval = {}

for sim_user in np.arange(0,500,50):
    top_poi_4_user_dict = defaultdict(list)
    for user in sample_users:
        top_poi_4_user_dict[user] = poi_recommend_4_user(user,user_weights_class,sim_user, 500)
    
    # Check Hit Radio Performance
    test_pairs_pos = orders_test.values.tolist()
    poi_hit = 0
    poi_num = 0
    for test_pair_index in sample_users:
        pos_pois = pos_pairs_dict_test[test_pair_index].keys()

        for poi in pos_pois:
            poi_num += 1
            if poi in top_poi_4_user_dict[test_pair_index]:
                poi_hit += 1

    hit_ratio = poi_hit/poi_num
    
    sim_user_eval[sim_user] = hit_ratio


# In[601]:


sim_user_eval


# In[458]:


# test to recommend to user
from collections import defaultdict

top_poi_4_user_dict = defaultdict(list)
for user in user_sampled_list:
    top_poi_4_user_dict[user] = poi_recommend_4_user(user,user_weights_class,500)


# In[ ]:


from collections import defaultdict

top_poi_4_user_dict = defaultdict(list)
for user in user_index:
    top_poi_4_user_dict[user] = poi_recommend_4_user(user,user_weights_class,500)


# In[449]:


# top_poi_4_user_dict


# In[450]:


orders_test.head()


# In[451]:


pos_pairs_list_test = orders_test[['user_id','wm_poi_id']].values.tolist()
pos_pairs_test = [tuple(link) for link in pos_pairs_list_test]
pos_pairs_set_test = set(pos_pairs_test)
pos_pairs_dict_test = defaultdict(dict)
for k,v in pos_pairs_set_test:
    dict2 = {}
    dict2[v] = 1
    pos_pairs_dict_test[k][v] = 1


# In[452]:


# Check Hit Radio Performance

test_pairs_pos = orders_test.values.tolist()

poi_hit = 0
poi_num = 0

for test_pair_index in user_sampled_list:
    pos_pois = pos_pairs_dict_test[test_pair_index].keys()
    
    for poi_index in pos_pois:
        poi_num += 1
        if poi_index in top_poi_4_user_dict[test_pair_index]:
            poi_hit += 1

hit_ratio = poi_hit/poi_num


# In[453]:


hit_ratio


# In[ ]:





# ## OUTPUT

# In[281]:


# output overall result
top_poi_4_user_values = []
for user in top_poi_4_user_dict:
    for rcmd in top_poi_4_user_dict[user]:
        top_poi_4_user_values.append([user,rcmd])

top_poi_4_user_df = pd.DataFrame(top_poi_4_user_values)


# In[271]:


# Sampling user result
path = './dat/order_train_sampling.csv'
user_sampled_list = pd.read_csv(path)['user_id'].unique()


# In[454]:


user_sampled_list


# In[272]:


# ouput sampling result
top_poi_4_user_values = []
for user in top_poi_4_user_dict:
    # output sample result
    if user in user_sampled_list:
        for rcmd in top_poi_4_user_dict[user]:
            top_poi_4_user_values.append([user,rcmd])
    else:
        continue

top_poi_4_user_sampled_df = pd.DataFrame(top_poi_4_user_values)
top_poi_4_user_sampled_df.rename(columns={0:'user_id',1:'wm_poi_id'},inplace=True)


# In[279]:


top_poi_4_user_sampled_df.to_csv("dat/top_poi_4_user_sampled.csv", index=False)


# In[280]:


top_poi_4_user_sampled_df.head()


# In[ ]:





# In[ ]:





# ## 2.2 Click Hsitory

# In[305]:


path = './原数据-20210301-20210328/orders_poi_session.txt'
click_his = pd.read_csv(path, sep = "\t")


# In[306]:


user_click_raw = pd.merge(orders_train, click_his, on='wm_order_id')[['user_id', 'clicks']]
user_click_raw


# In[308]:


user_click_df = user_click_raw.drop('clicks', axis=1).join(user_click_raw['clicks'].str.split('#', expand=True).stack().reset_index(level=1, drop=True).rename('click'))


# In[315]:


user_click_list = user_click_df.values.tolist()


# In[316]:


user_click_list



# ## 2.3 Geo Relation

user_poi_geo_list



# ## 2.4 Integrating the 3 Recall Method

# In[329]:


intg_poi_4_user_idct = top_poi_4_user_dict
intg_poi_4_user_idct = defaultdict(list)
for user_click in user_click_list:
    if user_click[1] not in intg_poi_4_user_idct[user_click[0]]:
        intg_poi_4_user_idct[user_click[0]].append(user_click[1])
for user_poi_geo_pair in user_poi_geo_list:
    if user_poi_geo_pair[1] not in intg_poi_4_user_idct[user_poi_geo_pair[0]]:
        intg_poi_4_user_idct[user_poi_geo_pair[0]].append(user_poi_geo_pair[1])

intg_poi_4_user_df = pd.DataFrame(intg_poi_4_user_idct)
intg_poi_4_user_df.rename(columns={0:'user_id',1:'wm_poi_id'},inplace=True)
intg_poi_4_user_df.to_csv("dat/top_poi_4_user_sampled.csv", index=False)

# In[332]:


# Check Hit Radio Performance

test_pairs_pos = orders_test.values.tolist()

poi_hit = 0
poi_num = 0

for test_pair_index in range(10000):
    pos_pois = pos_pairs_dict_test[test_pair_index].keys()
    
    for poi_index in pos_pois:
        poi_num += 1
        if poi_index in intg_poi_4_user_idct[test_pair_index]:
            poi_hit += 1

hit_ratio = poi_hit/poi_num


# In[337]:


poi_hit


# In[ ]:





# In[ ]:





# # 3 Geo Filtering

# In[564]:


pos_geo_pair = orders_train[['aor_id','aoi_id']]
pos_geo_pair = poi_user_pos_geo_pair.dropna(axis=0)
pos_geo_pair['aoi_id'] = poi_user_pos_geo_pair['aoi_id'].astype('int')
pos_geo_pair = pos_geo_pair.groupby(['aor_id','aoi_id'], as_index=False).size()


# In[565]:


pos_geo_pair_sort = pos_geo_pair.sort_values('size', ascending=False)




# In[566]:


pos_geo_pair_filtered = pos_geo_pair_sort[pos_geo_pair_sort['size'] >= 200]


# In[567]:


pos_geo_pair_filtered


# In[570]:


pos_geo_pair_valid = pos_geo_pair_filtered[['aor_id','aoi_id']].values.tolist()


# In[591]:


for user in intg_poi_4_user_idct.keys():
    for poi in intg_poi_4_user_idct[user]:
        poi_aor = int(10000 if pd.isnull(pois[pois['wm_poi_id']==poi]['aor_id']) else pois[pois['wm_poi_id']==poi]['aor_id'])
        user_aoi = int(orders_train[orders_train['user_id']==user]['aoi_id'][0])
        
        if [user, poi] not in pos_geo_pair_valid:
            del intg_poi_4_user_idct[user][poi]

