import pandas as pd
import csv
from pandas import ExcelWriter
import re
from datetime import datetime
import flask_excel as excel
from flask import Flask, render_template, redirect, request, url_for, session,  flash, send_file
from flask import Response
import nltk
from sklearn.metrics import confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import matplotlib.pyplot as plt
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from keras.models import load_model, Sequential
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
import io
import seaborn as sn
from keras.utils import pad_sequences 
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential 
from keras.layers import LSTM, Dense, Embedding

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def dashboard():
    return render_template("dashboard.html")

@app.route("/dashboard", methods=["POST", "GET"])
def dashboards():
    return render_template("dashboard.html") 

# @app.route("/UploadDataTrain", methods=["POST", "GET"])
# def Uploaddata():
#     if request.method == 'GET':
#         return render_template('prediksi.html')
#     elif request.method == 'POST':
        
#         plt.switch_backend('agg') 
#         excel_file = request.files["file"]        
        
#         # df =pd.read_excel(excel_file)
#         df_translate = pd.read_excel(excel_file)
#         df_translate
        
#         food_count  = df_translate['food'].apply(lambda x: x != '').sum()
#         place_count = df_translate['place'].apply(lambda x: x != '').sum()
#         service_count =df_translate['service'].apply(lambda x: x != '').sum()
#         price_count = df_translate['price'].apply(lambda x: x != '').sum()
#         print(f'food = {food_count}')
#         print(f'place = {place_count}')
#         print(f'service = {service_count}')
#         print(f'price = {price_count}')

#         import matplotlib.pyplot as plt

#         # Create a list of the counts
#         counts = [food_count, place_count, service_count, price_count]

#         # Create a list of the categories
#         categories = ['food', 'place', 'service', 'price']

#         # Create the bar chart
#         plt.bar(categories, counts)

#         # Add a title and axis labels
#         plt.title('Aspek Kategori')
#         plt.xlabel('kategori')
#         plt.ylabel('Jumlah')

#         # Show the chart
#         plt.show()



#         # create a figure with 2 rows and 2 columns of subplots
#         fig, axs = plt.subplots(2, 3, figsize=(15, 10))

#         counts = [food_count, place_count, service_count, price_count]
#         categories = ['food', 'place', 'service', 'price']

#         plt.subplot2grid((2,3), (0,0), rowspan = 3, colspan = 1).bar(categories, counts)

#         # plot the food sentiment counts in the first subplot
#         food_sentiment_counts = df_translate['food sentiment'].value_counts()
#         axs[0,1].bar(food_sentiment_counts.index, food_sentiment_counts.values)
#         axs[0,1].set_title('Food Sentiment Counts')
#         axs[0,1].set_xlabel('Sentiment')
#         axs[0,1].set_ylabel('Count')

#         # plot the place sentiment counts in the second subplot
#         place_sentiment_counts = df_translate['place sentiment'].value_counts()
#         axs[0,2].bar(place_sentiment_counts.index, place_sentiment_counts.values)
#         axs[0,2].set_title('Place Sentiment Counts')
#         axs[0,2].set_xlabel('Sentiment')
#         axs[0,2].set_ylabel('Count')

#         # plot the price sentiment counts in the third subplot
#         price_sentiment_counts = df_translate['price sentiment'].value_counts()
#         axs[1, 1].bar(price_sentiment_counts.index, price_sentiment_counts.values)
#         axs[1, 1].set_title('Price Sentiment Counts')
#         axs[1, 1].set_xlabel('Sentiment')
#         axs[1, 1].set_ylabel('Count')

#         # plot the service sentiment counts in the fourth subplot
#         service_sentiment_counts = df_translate['service sentiment'].value_counts()
#         axs[1, 2].bar(service_sentiment_counts.index, service_sentiment_counts.values)
#         axs[1, 2].set_title('Service Sentiment Counts')
#         axs[1, 2].set_xlabel('Sentiment')
#         axs[1, 2].set_ylabel('Count')

#         # show the figure
#         plt.show()

#         tb_counter_food = df_translate['food sentiment'].value_counts()
#         tb_counter_place = df_translate['place sentiment'].value_counts()
#         tb_counter_price = df_translate['price sentiment'].value_counts()
#         tb_counter_service = df_translate['service sentiment'].value_counts()

#         # Add a title and axis labels
#         plt.title('Aspek Kategori')
#         plt.xlabel('kategori')
#         plt.ylabel('Jumlah')

#         # Show the chart
#         plt.show()

#         # create a figure with 2 rows and 2 columns of subplots
#         fig, axs = plt.subplots(2, 2, figsize=(15, 10))

#         food_sentiment_counts = df_translate['food sentiment'].value_counts()
#         axs[0,0].pie(tb_counter_food.values, labels=tb_counter_food.index, )
#         axs[0,0].set_title('Food Sentiment Counts')
#         axs[0,0].set_xlabel('Sentiment')
#         axs[0,0].set_ylabel('Count')

#         # plot the place sentiment counts in the second subplot
#         place_sentiment_counts = df_translate['place sentiment'].value_counts()
#         axs[0,1].pie(tb_counter_place.values, labels=tb_counter_place.index, )
#         axs[0,1].set_title('Place Sentiment Counts')
#         axs[0,1].set_xlabel('Sentiment')
#         axs[0,1].set_ylabel('Count')

#         # plot the price sentiment counts in the third subplot
#         price_sentiment_counts = df_translate['price sentiment'].value_counts()
#         axs[1, 0].pie(tb_counter_price.values, labels=tb_counter_price.index, )
#         axs[1, 0].set_title('Price Sentiment Counts')
#         axs[1, 0].set_xlabel('Sentiment')
#         axs[1, 0].set_ylabel('Count')

#         # plot the service sentiment counts in the fourth subplot
#         service_sentiment_counts = df_translate['service sentiment'].value_counts()
#         axs[1, 1].pie(tb_counter_service.values, labels=tb_counter_service.index, )
#         axs[1, 1].set_title('Service Sentiment Counts')
#         axs[1, 1].set_xlabel('Sentiment')
#         axs[1, 1].set_ylabel('Count')

#         # show the figure
#         plt.show()

#         # image save

#         # Define the categories of sentiment
#         sentiment_categories = {
#             "negative": -1,
#             "half negative": -0.5,
#             "neutral": 0,
#             "half positive": 0.5,
#             "positive": 1,
#             np.nan: 0  # add a sentiment value for NaN
#         }

#         # Load your data into a Pandas DataFrame
#         df = df_translate

#         # Calculate the average sentiment score for each category
#         df["food sentiment score"] = df["food sentiment"].apply(lambda sentiment: sentiment_categories[sentiment])
#         df["place sentiment score"] = df["place sentiment"].apply(lambda sentiment: sentiment_categories[sentiment])
#         df["price sentiment score"] = df["price sentiment"].apply(lambda sentiment: sentiment_categories[sentiment])
#         df["service sentiment score"] = df["service sentiment"].apply(lambda sentiment: sentiment_categories[sentiment])
#         df["average sentiment score"] = df[["food sentiment score", "place sentiment score", "price sentiment score", "service sentiment score"]].mean(axis=1)

#         # Assign a sentiment label to each row based on the average sentiment score
#         df["sentiment label"] = df["average sentiment score"].apply(lambda score: "negative" if score < -0.5 else "half negative" if -0.5 <= score < 0 else "neutral" if pd.isna(score) else "half positive" if 0 < score <= 0.5 else "positive")

#         # Print the resulting DataFrame
#         df
#         # save image 


#         train_size = int(0.8 * len(df))
#         train_df = df[:train_size]
#         test_df = df[train_size:]

#         # Tokenize the text data
#         tokenizer = Tokenizer()
#         tokenizer.fit_on_texts(train_df["Lemma"])

#         # Convert the text data into sequences of numerical values
#         train_sequences = tokenizer.texts_to_sequences(train_df["Lemma"])
#         test_sequences = tokenizer.texts_to_sequences(test_df["Lemma"])

#         frame1 = train_df
        
        
#         # Define the sentiment label to integer mapping
#         sentiment_to_label = {"negative": 0, "half negative": 1, "neutral": 2, "half positive": 3, "positive": 4}

#         # Define the LSTM model architecture
#         vocab_size = len(tokenizer.word_index) + 1
#         max_length = max(len(seq) for seq in train_sequences)
#         embedding_size = 100
#         model = Sequential()
#         model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length))
#         model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
#         model.add(Dense(units=len(sentiment_to_label), activation="softmax"))



#         # Convert the sentiment labels to one-hot encoded vectors
#         train_labels = to_categorical(train_df["sentiment label"].map(sentiment_to_label).values)
#         test_labels = to_categorical(test_df["sentiment label"].map(sentiment_to_label).values)

#         # Pad the sequences to a fixed length
#         train_data = pad_sequences(sequences=train_sequences, maxlen=max_length)
#         test_data = pad_sequences(sequences=test_sequences, maxlen=max_length)
#         # Compile the model
#         model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
#         # Train the model
#         batch_size = 32
#         epochs = 10
#         history=model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_data, test_labels))

#         # Evaluate the model
#         loss, accuracy = model.evaluate(test_data, test_labels)
#         print("Test loss:", loss)
#         print("Test accuracy:", accuracy)


#         # Extract the accuracy and loss values from the history
#         acc = history.history['accuracy']
#         val_acc = history.history['val_accuracy']
#         loss = history.history['loss']
#         val_loss = history.history['val_loss']

#         # Plot the accuracy and loss curves
#         plt.figure()
#         plt.plot(acc, label='Training accuracy')
#         plt.plot(val_acc, label='Validation accuracy')
#         plt.legend()
#         plt.title('Accuracy')
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy')

#         plt.figure()
#         plt.plot(loss, label='Training loss')
#         plt.plot(val_loss, label='Validation loss')
#         plt.legend()
#         plt.title('Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
        
#         frame1.to_excel('frame1.xlsx',index=False)
#         # coba.to_excel('coba.xlsx')
#         # Foods=food_count
#         # Palace=place_count
#         # Service=service_count
#         # Price=price_count
#         # session["food"]=str(Foods)
#         # session["palace"]=str(Palace)
#         # session["service"]=str(Service)
#         # session["price"]=str(Price)
        
#         # url_for('grafik',Foods=Foods ,Palace=Palace,Service=Service,Price=Price)
#         return redirect(url_for('prediksi',))
    

# @app.route("/prediksi", methods=["POST", "GET"])
# def prediksi():
#     hasil1 = pd.read_excel('frame1.xlsx')
#     # Foods=session.get('food',None)
#     # Palace=session.get('palace',None)
#     # Service=session.get('service',None)
#     # Price=session.get('price',None)
#     # hasil2 = pd.read_excel('coba.xlsx')    
#     return (render_template("Result_Prediksi.html", tables=[hasil1.to_html(classes='data table table-bordered table-striped',justify='center',index=False,show_dimensions=True,max_rows=1500).replace('\n','').replace('[""]','').replace("<thead>", "<thead class='thead-primary table-hover'>").replace('Unnamed: 0',"No")],titles=hasil1.columns.values,
#                             # cobaaja=hasil2.to_html(classes='data',justify='center')
#                             # Foods=Foods ,Palace=Palace,Service=Service,Price=Price
#                             ))



@app.route("/prediksiTable", methods=["POST", "GET"])
def Table():
    return render_template('Result_Prediksi.html')

@app.route("/UpTrain", methods=["POST", "GET"])
def UpTrain():
    if request.method == 'GET':
        return render_template('Newup.html')
    elif request.method == 'POST':
        # plt.switch_backend('agg')
        excel_file = request.files["file"]        
        # df_translate = pd.read_excel('dataScapingTripadvisorFIXXTranslate.xlsx')
        df_translate =pd.read_excel(excel_file)
        df_translate.to_excel("dataTrain.xlsx",index=False)
        df_translate
        
        food_count  = df_translate['food'].apply(lambda x: x != '').sum()
        place_count = df_translate['place'].apply(lambda x: x != '').sum()
        service_count =df_translate['service'].apply(lambda x: x != '').sum()
        price_count = df_translate['price'].apply(lambda x: x != '').sum()
        print(f'food = {food_count}')
        print(f'place = {place_count}')
        print(f'service = {service_count}')
        print(f'price = {price_count}')

        
        # Create a list of the counts
        counts = [food_count, place_count, service_count, price_count]

        # Create a list of the categories
        categories = ['food', 'place', 'service', 'price']

        # Create the bar chart
        plt.bar(categories, counts)

        # Add a title and axis labels
        plt.title('Aspek Kategori')
        plt.xlabel('kategori')
        plt.ylabel('Jumlah')

        # Show the chart
        #plt.show()
        
        # create a figure with 2 rows and 2 columns of subplots
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        counts = [food_count, place_count, service_count, price_count]
        categories = ['food', 'place', 'service', 'price']

        plt.subplot2grid((2,3), (0,0), rowspan = 3, colspan = 1).bar(categories, counts)

        # plot the food sentiment counts in the first subplot
        food_sentiment_counts = df_translate['food sentiment'].value_counts()
        axs[0,1].bar(food_sentiment_counts.index, food_sentiment_counts.values)
        axs[0,1].set_title('Food Sentiment Counts')
        axs[0,1].set_xlabel('Sentiment')
        axs[0,1].set_ylabel('Count')

        # plot the place sentiment counts in the second subplot
        place_sentiment_counts = df_translate['place sentiment'].value_counts()
        axs[0,2].bar(place_sentiment_counts.index, place_sentiment_counts.values)
        axs[0,2].set_title('Place Sentiment Counts')
        axs[0,2].set_xlabel('Sentiment')
        axs[0,2].set_ylabel('Count')

        # plot the price sentiment counts in the third subplot
        price_sentiment_counts = df_translate['price sentiment'].value_counts()
        axs[1, 1].bar(price_sentiment_counts.index, price_sentiment_counts.values)
        axs[1, 1].set_title('Price Sentiment Counts')
        axs[1, 1].set_xlabel('Sentiment')
        axs[1, 1].set_ylabel('Count')

        # plot the service sentiment counts in the fourth subplot
        service_sentiment_counts = df_translate['service sentiment'].value_counts()
        axs[1, 2].bar(service_sentiment_counts.index, service_sentiment_counts.values)
        axs[1, 2].set_title('Service Sentiment Counts')
        axs[1, 2].set_xlabel('Sentiment')
        axs[1, 2].set_ylabel('Count')

        # show the figure
        #plt.show()
        plt.savefig('static/img/BoxCharts.png', bbox_inches="tight")
        
        
        
        tb_counter_food = df_translate['food sentiment'].value_counts()
        tb_counter_place = df_translate['place sentiment'].value_counts()
        tb_counter_price = df_translate['price sentiment'].value_counts()
        tb_counter_service = df_translate['service sentiment'].value_counts()
        
        # create a figure with 2 rows and 2 columns of subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        food_sentiment_counts = df_translate['food sentiment'].value_counts()
        axs[0,0].pie(tb_counter_food.values, labels=tb_counter_food.index, )
        axs[0,0].set_title('Food Sentiment Counts')
        axs[0,0].set_xlabel('Sentiment')
        axs[0,0].set_ylabel('Count')

        # plot the place sentiment counts in the second subplot
        place_sentiment_counts = df_translate['place sentiment'].value_counts()
        axs[0,1].pie(tb_counter_place.values, labels=tb_counter_place.index, )
        axs[0,1].set_title('Place Sentiment Counts')
        axs[0,1].set_xlabel('Sentiment')
        axs[0,1].set_ylabel('Count')

        # plot the price sentiment counts in the third subplot
        price_sentiment_counts = df_translate['price sentiment'].value_counts()
        axs[1, 0].pie(tb_counter_price.values, labels=tb_counter_price.index, )
        axs[1, 0].set_title('Price Sentiment Counts')
        axs[1, 0].set_xlabel('Sentiment')
        axs[1, 0].set_ylabel('Count')

        # plot the service sentiment counts in the fourth subplot
        service_sentiment_counts = df_translate['service sentiment'].value_counts()
        axs[1, 1].pie(tb_counter_service.values, labels=tb_counter_service.index, )
        axs[1, 1].set_title('Service Sentiment Counts')
        axs[1, 1].set_xlabel('Sentiment')
        axs[1, 1].set_ylabel('Count')

        # show the figure
        #plt.show()
        plt.savefig('static/img/PlotsCharts.png', bbox_inches="tight")
        
        # Define the categories of sentiment
        sentiment_categories = {
            "negative": -1,
            "half negative": -0.5,
            "neutral": 0,
            "half positive": 0.5,
            "positive": 1,
            np.nan: 0  # add a sentiment value for NaN
        }

        # Load your data into a Pandas DataFrame
        df = df_translate

        # Calculate the average sentiment score for each category
        df["food sentiment score"] = df["food sentiment"].apply(lambda sentiment: sentiment_categories[sentiment])
        df["place sentiment score"] = df["place sentiment"].apply(lambda sentiment: sentiment_categories[sentiment])
        df["price sentiment score"] = df["price sentiment"].apply(lambda sentiment: sentiment_categories[sentiment])
        df["service sentiment score"] = df["service sentiment"].apply(lambda sentiment: sentiment_categories[sentiment])
        df["average sentiment score"] = df[["food sentiment score", "place sentiment score", "price sentiment score", "service sentiment score"]].mean(axis=1)

        # Assign a sentiment label to each row based on the average sentiment score
        df["sentiment label"] = df["average sentiment score"].apply(lambda score: "negative" if score < -0.5 else "half negative" if -0.5 <= score < 0 else "neutral" if pd.isna(score) else "half positive" if 0 < score <= 0.5 else "positive")

        # Print the resulting DataFrame
        df
        
        train_size = int(0.8 * len(df))
        train_df = df[:train_size]
        test_df = df[train_size:]

        # Tokenize the text data
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_df["Lemma"])

        # Convert the text data into sequences of numerical values
        train_sequences = tokenizer.texts_to_sequences(train_df["Lemma"])
        test_sequences = tokenizer.texts_to_sequences(test_df["Lemma"])

        hasil1 = train_df
        
        
        
        # Define the sentiment label to integer mapping
        sentiment_to_label = {"negative": 0, "half negative": 1, "neutral": 2, "half positive": 3, "positive": 4}

        # Define the LSTM model architecture
        vocab_size = len(tokenizer.word_index) + 1
        max_length = max(len(seq) for seq in train_sequences)
        embedding_size = 100
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length))
        model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(units=len(sentiment_to_label), activation="softmax"))



        # Convert the sentiment labels to one-hot encoded vectors
        train_labels = to_categorical(train_df["sentiment label"].map(sentiment_to_label).values)
        test_labels = to_categorical(test_df["sentiment label"].map(sentiment_to_label).values)

        # Pad the sequences to a fixed length
        train_data = pad_sequences(sequences=train_sequences, maxlen=max_length)
        test_data = pad_sequences(sequences=test_sequences, maxlen=max_length)
        # Compile the model
        model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
        # Train the model
        batch_size = 32
        epochs = 10
        history=model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_data, test_labels))


        # Evaluate the model
        loss, accuracy = model.evaluate(test_data, test_labels)
        print("Test loss:", loss)
        print("Test accuracy:", accuracy)



        
        
        
        # Extract the accuracy and loss values from the history
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # Plot the accuracy and loss curves
        plt.figure()
        plt.plot(acc, label='Training accuracy')
        plt.plot(val_acc, label='Validation accuracy')
        plt.legend()
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        #plt.show()
        plt.savefig('static/img/chartAcc.png', bbox_inches="tight")
        

        plt.figure()
        plt.plot(loss, label='Training loss')
        plt.plot(val_loss, label='Validation loss')
        plt.legend()
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        #plt.show()
        plt.savefig('static/img/chartLos.png', bbox_inches="tight")
        
        # Get the predicted sentiment labels
        predicted_labels = np.argmax(model.predict(test_data), axis=1)

        # Get the true sentiment labels
        true_labels = np.argmax(test_labels, axis=1)

        # Create the confusion matrix
        conf_matrix = confusion_matrix(true_labels, predicted_labels)

        # Print the confusion matrix
        print("Confusion matrix:")
        print(conf_matrix)
        
        # Get the predicted probabilities for the test data
        y_pred_prob = model.predict(test_data)

        # Convert predicted probabilities to predicted labels
        y_pred = np.argmax(y_pred_prob, axis=1)

        # Get the true labels for the test data
        y_true = np.argmax(test_labels, axis=1)

        # Compute confusion matrix
        confusion_mat = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        plt.matshow(confusion_mat)
        plt.colorbar()
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        #plt.show()
        plt.savefig('static/img/matrix.png', bbox_inches="tight")
        
        return (render_template("Training.html",tables=[hasil1.to_html(classes='data table table-bordered table-striped',justify='center',index=False,show_dimensions=True,max_rows=1500).replace('\n','').replace('[""]','').replace("<thead>", "<thead class='thead-primary table-hover'>").replace('Unnamed: 0',"No")],titles=hasil1.columns.values))


@app.route("/grafik", methods=["POST", "GET"])
def Grafik():
    
    return render_template('grafik.html')



@app.route("/UpdataTrain", methods=["POST", "GET"])
def Training():
    if request.method == 'GET':
        return render_template('Updata.html')
    elif request.method == 'POST':
        #plt.switch_backend('agg')
        df_translate= pd.read_excel('dataTrain.xlsx')
        df_translate
        
        food_count  = df_translate['food'].apply(lambda x: x != '').sum()
        place_count = df_translate['place'].apply(lambda x: x != '').sum()
        service_count =df_translate['service'].apply(lambda x: x != '').sum()
        price_count = df_translate['price'].apply(lambda x: x != '').sum()
        print(f'food = {food_count}')
        print(f'place = {place_count}')
        print(f'service = {service_count}')
        print(f'price = {price_count}')

        
        # Create a list of the counts
        counts = [food_count, place_count, service_count, price_count]

        # Create a list of the categories
        categories = ['food', 'place', 'service', 'price']

        # Create the bar chart
        plt.bar(categories, counts)

        # Add a title and axis labels
        plt.title('Aspek Kategori')
        plt.xlabel('kategori')
        plt.ylabel('Jumlah')

        # Show the chart
        #plt.show()
        
        # create a figure with 2 rows and 2 columns of subplots
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        counts = [food_count, place_count, service_count, price_count]
        categories = ['food', 'place', 'service', 'price']

        plt.subplot2grid((2,3), (0,0), rowspan = 3, colspan = 1).bar(categories, counts)

        # plot the food sentiment counts in the first subplot
        food_sentiment_counts = df_translate['food sentiment'].value_counts()
        axs[0,1].bar(food_sentiment_counts.index, food_sentiment_counts.values)
        axs[0,1].set_title('Food Sentiment Counts')
        axs[0,1].set_xlabel('Sentiment')
        axs[0,1].set_ylabel('Count')

        # plot the place sentiment counts in the second subplot
        place_sentiment_counts = df_translate['place sentiment'].value_counts()
        axs[0,2].bar(place_sentiment_counts.index, place_sentiment_counts.values)
        axs[0,2].set_title('Place Sentiment Counts')
        axs[0,2].set_xlabel('Sentiment')
        axs[0,2].set_ylabel('Count')

        # plot the price sentiment counts in the third subplot
        price_sentiment_counts = df_translate['price sentiment'].value_counts()
        axs[1, 1].bar(price_sentiment_counts.index, price_sentiment_counts.values)
        axs[1, 1].set_title('Price Sentiment Counts')
        axs[1, 1].set_xlabel('Sentiment')
        axs[1, 1].set_ylabel('Count')

        # plot the service sentiment counts in the fourth subplot
        service_sentiment_counts = df_translate['service sentiment'].value_counts()
        axs[1, 2].bar(service_sentiment_counts.index, service_sentiment_counts.values)
        axs[1, 2].set_title('Service Sentiment Counts')
        axs[1, 2].set_xlabel('Sentiment')
        axs[1, 2].set_ylabel('Count')

        # show the figure
        #plt.show()
        
        
        
        tb_counter_food = df_translate['food sentiment'].value_counts()
        tb_counter_place = df_translate['place sentiment'].value_counts()
        tb_counter_price = df_translate['price sentiment'].value_counts()
        tb_counter_service = df_translate['service sentiment'].value_counts()
        
        # create a figure with 2 rows and 2 columns of subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        food_sentiment_counts = df_translate['food sentiment'].value_counts()
        axs[0,0].pie(tb_counter_food.values, labels=tb_counter_food.index, )
        axs[0,0].set_title('Food Sentiment Counts')
        axs[0,0].set_xlabel('Sentiment')
        axs[0,0].set_ylabel('Count')

        # plot the place sentiment counts in the second subplot
        place_sentiment_counts = df_translate['place sentiment'].value_counts()
        axs[0,1].pie(tb_counter_place.values, labels=tb_counter_place.index, )
        axs[0,1].set_title('Place Sentiment Counts')
        axs[0,1].set_xlabel('Sentiment')
        axs[0,1].set_ylabel('Count')

        # plot the price sentiment counts in the third subplot
        price_sentiment_counts = df_translate['price sentiment'].value_counts()
        axs[1, 0].pie(tb_counter_price.values, labels=tb_counter_price.index, )
        axs[1, 0].set_title('Price Sentiment Counts')
        axs[1, 0].set_xlabel('Sentiment')
        axs[1, 0].set_ylabel('Count')

        # plot the service sentiment counts in the fourth subplot
        service_sentiment_counts = df_translate['service sentiment'].value_counts()
        axs[1, 1].pie(tb_counter_service.values, labels=tb_counter_service.index, )
        axs[1, 1].set_title('Service Sentiment Counts')
        axs[1, 1].set_xlabel('Sentiment')
        axs[1, 1].set_ylabel('Count')

        # show the figure
        #plt.show()
        plt.savefig('static/img/PlotsCharts.png', bbox_inches="tight")
        
        # Define the categories of sentiment
        sentiment_categories = {
            "negative": -1,
            "half negative": -0.5,
            "neutral": 0,
            "half positive": 0.5,
            "positive": 1,
            np.nan: 0  # add a sentiment value for NaN
        }

        # Load your data into a Pandas DataFrame
        df = df_translate

        # Calculate the average sentiment score for each category
        df["food sentiment score"] = df["food sentiment"].apply(lambda sentiment: sentiment_categories[sentiment])
        df["place sentiment score"] = df["place sentiment"].apply(lambda sentiment: sentiment_categories[sentiment])
        df["price sentiment score"] = df["price sentiment"].apply(lambda sentiment: sentiment_categories[sentiment])
        df["service sentiment score"] = df["service sentiment"].apply(lambda sentiment: sentiment_categories[sentiment])
        df["average sentiment score"] = df[["food sentiment score", "place sentiment score", "price sentiment score", "service sentiment score"]].mean(axis=1)

        # Assign a sentiment label to each row based on the average sentiment score
        df["sentiment label"] = df["average sentiment score"].apply(lambda score: "negative" if score < -0.5 else "half negative" if -0.5 <= score < 0 else "neutral" if pd.isna(score) else "half positive" if 0 < score <= 0.5 else "positive")

        # Print the resulting DataFrame
        df
        
        train_size = int(0.8 * len(df))
        train_df = df[:train_size]
        test_df = df[train_size:]

        # Tokenize the text data
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_df["Lemma"])

        # Convert the text data into sequences of numerical values
        train_sequences = tokenizer.texts_to_sequences(train_df["Lemma"])
        test_sequences = tokenizer.texts_to_sequences(test_df["Lemma"])

        hasil1 = train_df
        
        
        
        # Define the sentiment label to integer mapping
        sentiment_to_label = {"negative": 0, "half negative": 1, "neutral": 2, "half positive": 3, "positive": 4}

        # Define the LSTM model architecture
        vocab_size = len(tokenizer.word_index) + 1
        max_length = max(len(seq) for seq in train_sequences)
        embedding_size = 100
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length))
        model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(units=len(sentiment_to_label), activation="softmax"))



        # Convert the sentiment labels to one-hot encoded vectors
        train_labels = to_categorical(train_df["sentiment label"].map(sentiment_to_label).values)
        test_labels = to_categorical(test_df["sentiment label"].map(sentiment_to_label).values)

        # Pad the sequences to a fixed length
        train_data = pad_sequences(sequences=train_sequences, maxlen=max_length)
        test_data = pad_sequences(sequences=test_sequences, maxlen=max_length)
        # Compile the model
        model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
        # Train the model
        batch_size = 32
        epochs = 10
        history=model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_data, test_labels))


        # Evaluate the model
        loss, accuracy = model.evaluate(test_data, test_labels)
        print("Test loss:", loss)
        print("Test accuracy:", accuracy)



        
        
        
        # Extract the accuracy and loss values from the history
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # Plot the accuracy and loss curves
        plt.figure()
        plt.plot(acc, label='Training accuracy')
        plt.plot(val_acc, label='Validation accuracy')
        plt.legend()
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        

        plt.figure()
        plt.plot(loss, label='Training loss')
        plt.plot(val_loss, label='Validation loss')
        plt.legend()
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # Get the predicted sentiment labels
        predicted_labels = np.argmax(model.predict(test_data), axis=1)

        # Get the true sentiment labels
        true_labels = np.argmax(test_labels, axis=1)

        # Create the confusion matrix
        conf_matrix = confusion_matrix(true_labels, predicted_labels)

        # Print the confusion matrix
        print("Confusion matrix:")
        print(conf_matrix)
        
        # Get the predicted probabilities for the test data
        y_pred_prob = model.predict(test_data)

        # Convert predicted probabilities to predicted labels
        y_pred = np.argmax(y_pred_prob, axis=1)

        # Get the true labels for the test data
        y_true = np.argmax(test_labels, axis=1)

        # Compute confusion matrix
        confusion_mat = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        plt.matshow(confusion_mat)
        plt.colorbar()
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        # plt.show()
        
        excel_file = request.files["file"]
        data_predict =pd.read_excel(excel_file)
        data_predict
        df=data_predict
        # Define the label-to-sentiment dictionary
        label_to_sentiment = {0: 'negative', 1: 'half negative', 2: 'neutral', 3: 'half positive', 4: 'positive'}

        # Create a new column to store the predicted sentiment labels
        df["predicted_sentiment"] = ""

        # Preprocess the data
        sequences = tokenizer.texts_to_sequences(data_predict["Translate"])
        data = pad_sequences(sequences, maxlen=max_length)

        # Make predictions on the data and store the predicted sentiment labels in the new column
        for i in range(len(data)):
            prediction = model.predict(data[i:i+1])[0]
            sentiment_label = np.argmax(prediction)
            sentiment = label_to_sentiment[sentiment_label]
            df.at[i, "predicted_sentiment"] = sentiment

        # Print the dataframe with the predicted sentiment labels
        print(df.head())
        frames= df

            
    return (render_template("Hasil.html",tables=[frames.to_html(classes='data table table-bordered table-striped',justify='center',show_dimensions=True,index=False)],titles=frames.columns.values))

if __name__ == "__main__":
    app.secret_key = "ss"
    app.run(debug=True)