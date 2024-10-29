import run_inference
import train_model

train_model.train_yolo_model(config = "/MVPCD/config/config.yaml",epochs=50, learning_rate=0.0001, batch_size=4, weight_decay = 0.0005, amp=True, freeze=["backbone"])
train_model.train_yolo_model(config = "/MVPCD/config/config.yaml",epochs=50, learning_rate=0.0001, batch_size=4, weight_decay = 0.0005, amp=True)
train_model.train_yolo_model(config = "/MVPCD/config/config.yaml",epochs=50, learning_rate=0.0001, batch_size=4, weight_decay = 0.0005, freeze=["backbone"])
train_model.train_yolo_model(config = "/MVPCD/config/config.yaml",epochs=50, learning_rate=0.0001, batch_size=4, weight_decay = 0.0005)
train_model.train_yolo_model(config = "/MVPCD/config/config.yaml",epochs=50, learning_rate=0.0001, batch_size=4, amp=True, freeze=["backbone"])
train_model.train_yolo_model(config = "/MVPCD/config/config.yaml",epochs=50, learning_rate=0.0001, batch_size=4, amp=True)
train_model.train_yolo_model(config = "/MVPCD/config/config.yaml",epochs=50, learning_rate=0.0001, batch_size=4, freeze=["backbone"])










        # weight_decay=0.0005,  # Add regularization
        # amp=True,    # Enable mixed precision if supported
        # freeze=["backbone"] # Freeze the backbone layers

        # config, epochs=100, learning_rate=0.0001, batch_size=8