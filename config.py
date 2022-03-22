import torch


all_classes_modelnet40 = ["airplane", "bench", "bowl", "cone", "desk", "flower_pot", "keyboard", "mantel", "person", "radio",
                          "sofa", "table", "tv_stand", "xbox", "bathtub", "bookshelf", "car", "cup", "door", "glass_box"
                          "lamp", "monitor", "piano", "range_hood", "stairs", "tent", "vase", "bed", "bottle", "chair", "curtain",
                          "dresser", "guitar", "laptop", "night_stand", "plant", "sink", "stool", "toilet", "wardrobe"]



def get_config():
    return {
        "batch_size":8,
        "train_classes": all_classes_modelnet40,
        "backend_network": "baseline",
        "learning_rate": 3e-4,
        "optimizer":"adam",
    }


if __name__ == '__main__':
    config = get_config()
