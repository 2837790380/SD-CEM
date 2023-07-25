from dataset.util import *
import torch
import tqdm

def getCategoryVenueSet(checkinPath):
    categoryVenueSet = {}
    with open(checkinPath, encoding='utf-8') as file:
        for row in csv.reader(file):
            for item in row:
                venue, category = item.split("@")
                if categoryVenueSet.get(category) == None:
                    categoryVenueSet[category] = [venue]
                else:
                    categoryVenueSet[category].append(venue)
    file.close()
    for key, value in categoryVenueSet.items():
        categoryVenueSet[key] = set(value)
    return categoryVenueSet

def getVenueCoordinateMap(POI_path):
    with open(POI_path, encoding='utf-8') as file:
        content = file.readlines()
        file.close()

    JP_venueCoordinateMap = dict()
    US_venueCoordinateMap = dict()
    other_venueCoordinateMap = dict()
    for line in content:
        temp = line[:-1].split('\t')
        if temp[-1] == 'US':
            US_venueCoordinateMap[temp[0]] = (float(temp[1]), float(temp[2]))
        elif temp[-1] == 'JP':
            JP_venueCoordinateMap[temp[0]] = (float(temp[1]), float(temp[2]))
        else:
            other_venueCoordinateMap[temp[0]] = (float(temp[1]), float(temp[2]))

    return JP_venueCoordinateMap, US_venueCoordinateMap, other_venueCoordinateMap

def getUserVectorAndCoordinate(user_checkin, venueCoordinateMap, category_embedding):
    counts = len(user_checkin)

    category_ids, venue_ids = [], []
    for item in user_checkin:
        venue_id, category_id = item.split('@')
        category_ids.append(category_id)
        venue_ids.append(venue_id)

    category_embeddings = 0
    for item in category_ids:
        category_embeddings += category_embedding[item]

    user_Vector = category_embeddings / counts

    venueCoordinate = [venueCoordinateMap[item] for item in venue_ids]
    latitude, longitude = 0, 0
    for item in venueCoordinate:
        latitude += item[0]
        longitude += item[1]
    user_coordinate = (latitude / counts, longitude / counts)

    return user_Vector, user_coordinate


def findUserPreferedCategories(user_vector, query_ids, query_embedding, top_k):
    category_size = len(query_embedding)
    sim = np.zeros(category_size)
    for i in range(category_size):
        sim[i] = torch.nn.functional.cosine_similarity(query_embedding[i].clone().detach(),
                                                       user_vector, dim=0)

    count = 0
    user_preferdCategory = []
    # sorted_ids = (-1 * sim).argsort
    for i in (-1 * sim).argsort():
        count += 1
        user_preferdCategory.append(query_ids[i])

        if count >= top_k:
            return user_preferdCategory


import math

def rad(d):
    import math
    return math.pi * d / 180.0

def getDistanceBetweenLongitudeAndLatitude(o_lat, o_lon, n_lat, n_lon):
    earth_radius = 6378.137
    rad_lat1 = rad(o_lat)
    rad_lat2 = rad(n_lat)
    a = rad_lat1 - rad_lat2
    b = rad(o_lon) - rad(n_lon)

    s = 2 * math.asin(math.sqrt(
        math.pow(math.sin(a / 2), 2) + math.cos(rad_lat1) * math.cos(rad_lat2) * math.pow(math.sin(b / 2), 2)
    ))

    s = s * earth_radius

    return s

def get_nameAndId(categories_path):
    df = pd.read_csv(categories_path).values
    name_to_id = {}
    id_to_name = {}

    for item in df:
        name_to_id[item[1]] = item[0]
        id_to_name[item[0]] = item[1]

    return name_to_id, id_to_name


if __name__ == '__main__':
    categories_path = '../dataset/data/category.csv'
    embed_file_path = '../embeddings/SD-CEM#US#50.csv'
    poi_path = '../dataset/data/dataset_TIST2015_POIs.txt'
    checkin_path = '../dataset/data/CheckinLocationCategoryIDSequenceUS50Filter.csv'

    name_to_id, id_to_name = get_nameAndId(categories_path)
    with open(checkin_path, encoding='utf-8') as file:
        user_checkin = []
        for row in csv.reader(file):
            user_checkin += [row]
    file.close()

    categoryVenueSet = getCategoryVenueSet(checkinPath=checkin_path)

    JP_venueCoordinateMap, US_venueCoordinateMap, other_venueCoordinateMap = getVenueCoordinateMap(poi_path)

    embeddings = {}
    with open(embed_file_path) as f:
        for row in csv.reader(f):
            embeddings[name_to_id[row[0]]] = torch.Tensor(list(map(float, row[1:])))
    embed_size = len(list(embeddings.values())[0])

    a1, a2, b1, b2 = 1, 10, 1, 1

    p1, p5, p10 = 0, 0, 0
    r1, r5, r10 = 0, 0, 0

    uss = user_checkin

    for uc in tqdm.tqdm(uss):
        venueSetGroundtruth = set([item.split('@')[0] for item in uc])

        user_Vector, user_coordinate = getUserVectorAndCoordinate(uc, US_venueCoordinateMap, embeddings)

        preferedCategory = findUserPreferedCategories(user_Vector, list(embeddings.keys()), list(embeddings.values()), top_k=15)

        user_latitude, user_longitude = user_coordinate

        recommendList = []
        r_sort = []
        for i, category in enumerate(preferedCategory):
            if categoryVenueSet.get(category):
                venueSet = categoryVenueSet[category]
                for venue in venueSet:
                    recommendList.append(venue)
                    venue_latitude, venue_longitude = US_venueCoordinateMap[venue]
                    dist = getDistanceBetweenLongitudeAndLatitude(user_latitude, user_longitude, venue_latitude,
                                                                  venue_longitude)
                    # print(dist)
                    sim = (np.exp(-a2 * i)) * (np.exp(-b2 * dist))
                    r_sort.append(sim)

        r_index = np.argsort(r_sort)[::-1][:1]
        recommendSet = list(np.array(recommendList)[r_index])
        counts = set(recommendSet).intersection(set(venueSetGroundtruth))

        p1 = p1 + len(counts) / 1
        r1 = r1 + len(counts) / len(venueSetGroundtruth)

        r_index = np.argsort(r_sort)[::-1][:5]
        recommendSet = list(np.array(recommendList)[r_index])
        counts = set(recommendSet).intersection(set(venueSetGroundtruth))

        p5 = p5 + len(counts) / 5
        r5 = r5 + len(counts) / len(venueSetGroundtruth)

        r_index = np.argsort(r_sort)[::-1][:10]
        recommendSet = list(np.array(recommendList)[r_index])
        counts = set(recommendSet).intersection(set(venueSetGroundtruth))

        p10 = p10 + len(counts) / 10
        r10 = r10 + len(counts) / len(venueSetGroundtruth)

    print('-------------')
    print(f'precision@{1}', p1 / len(uss))
    print(f'recall@{1}', r1 / len(uss))
    print('-------------')
    print(f'precision@{5}', p5 / len(uss))
    print(f'recall@{5}', r5 / len(uss))
    print('-------------')
    print(f'precision@{10}', p10 / len(uss))
    print(f'recall@{10}', r10 / len(uss))
