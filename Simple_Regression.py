# import the python libraries
import pandas as pd  # use perform dataset analysis
import matplotlib.pyplot as plt  # use for creating plots
import matplotlib.ticker as y_ticker  # use to format the y-axis value to display values in millions
from matplotlib.widgets import Slider  # use to create a slider to help user choose any input variable
import numpy as np  # use to perform mathematical functions
from sklearn.linear_model import LinearRegression  # use to create and train the model
from sklearn.model_selection import train_test_split  # use to split data use for training and testing
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # use to evaluate models performance
import seaborn as sns  # use for data visualization

# declared variables to use
plotCount: int = 0  # use for track slider movement to clear previous plot
line_1 = (0, 0)  # stores the last slider vertical line plot
line_2 = (0, 0)  # stores the last slider vertical line plot
#
# DATA PRE-PROCESSING, MODEL BUILDING, AND MODEL EVALUATION
#
# read dataset csv file
df = pd.read_csv('houseprice_data.csv')
# remove any null values from data rows
df = df.dropna()
# get the independent (input) variable
X = df[['sqft_living']].values  # or you can use: X = df[:, 3].values
# get the dependent (target) variable
y = df[['price']].values  # or you can use: y = df[:, 0].values
# split data into test and train using 30% of data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=0)
# build the linear regression model
model = LinearRegression()
# train the linear regression model
model.fit(X_train, y_train)
# find the coefficients
coef_value = model.coef_
print('Coefficients: ', coef_value)
# find the intercept
intercept = model.intercept_
print('Intercept: ', intercept)
# find the mean squared errors
print('Mean squared error (Training): %.1f' % mean_squared_error(y_train, model.predict(X_train)))
print('Mean squared error (Testing): %.1f' % mean_squared_error(y_test, model.predict(X_test)))
# find the mean absolute error
print('Mean absolute error: ', round(mean_absolute_error(y_test, model.predict(X_test)), 1))
# find the R^2 value:
coef_det = r2_score(y_test, model.predict(X_test))
print('Coefficient of determination (R^2 score): ', round(r2_score(y_test, model.predict(X_test)) * 100, 2), '%')
# display the training and testing scores
print('Training Score: ', round(model.score(X_train, y_train) * 100, 1), '%')
print('Testing Score: ', round(model.score(X_test, y_test) * 100, 1), '%')
#
# GET THE MODEL EVALUATIONS AS A TEXT
#
# string all the model performance text with each on a newline
result_text = 'Evaluating the simple regression model: \n' \
              + 'Coefficients: ' + ' ,'.join(str(round(e[0], 1)) for e in coef_value) + '\n' \
              + 'Intercept: ' + ' ,'.join(str(round(e, 1)) for e in intercept) + '\n' \
              + str('Mean squared error (Training): %.1f' % mean_squared_error(y_train, model.predict(X_train))) + '\n' \
              + str('Mean squared error (Testing): %.1f' % mean_squared_error(y_test, model.predict(X_test))) + '\n' \
              + str('Mean absolute error: %.1f' % mean_absolute_error(y_test, model.predict(X_test))) + '\n' \
              + str('Coefficient of determination (R^2 score): '
                    + str(round(r2_score(y_test, model.predict(X_test)) * 100, 2)) + ' %' + '\n'
                    + 'Model training Score: ' + str(round(model.score(X_train, y_train) * 100, 1)) + ' %' + '\n'
                    + 'Model testing Score: ' + str(round(model.score(X_test, y_test) * 100, 1)) + ' %')
# create the new figure with width = 15 and height = 7
fig = plt.figure(figsize=(15, 7))
# display figure suptitle
fig.suptitle('Simple Regression By Umolu John Chukwuemeka (2065655)', color='blue', fontsize=16, fontweight='bold')
# create box to contain the model performance text using the matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# set the figure title
fig.canvas.manager.set_window_title('Coursework Task 1: Simple Regression By Umolu John Chukwuemeka (2065655)')
# create a subplot with a single figure
axa = fig.add_subplot(121)
# plot the scatter plot of x and y variables
axa.scatter(X, y, color='blue')
# draw the regression line
axa.plot(X, model.predict(X), color='red', label=' Predicted Regression line')
# set the x-axis label
axa.set_xlabel('x-axis (Square foot of the house)')
# set the y-axis label
axa.set_ylabel('y-axis (Price of the house)')
# set the minimum and maximum range for x-axis
axa.set_xlim(0, np.max(np.array(X)))
# set the minimum and maximum range for y-axis
axa.set_ylim(0, np.max(np.array(y)) + 1000000)
# format axis to display value in millions
# https://stackoverflow.com/questions/63723514/userwarning-fixedformatter-should-only-be-used-together-with-fixedlocator
ticks_loc = axa.get_yticks().tolist()
label_format = '{:,}'
axa.yaxis.set_major_locator(y_ticker.FixedLocator(ticks_loc))
axa.set_yticklabels([label_format.format(x) for x in ticks_loc])

# move the figure up and sideways so that it's not on top of the slider
fig.subplots_adjust(left=0.2, bottom=0.42)


# function used to predict values
def predict_price(value):
    # make the variables accessible from outside the function
    global input_variables, prediction, display_text
    # get slider value and convert it to numpy array
    input_variables = np.array([[value]])
    # get the predicted price value using the built model
    prediction = model.predict(input_variables)
    # variable to store all the square foot and prediction text to be displayed
    display_text = 'Square foot of the house: ' + str(value) + '\n' \
                   + 'Predicted Price of the house: ' + str(round(prediction[0][0], 1))
    # display the square foot and prediction text using figure subtitle
    axa.set_title(display_text, fontsize=12, fontweight='bold')
    # display the square foot and prediction text using figure subtitle on console
    print(display_text)
    # return the model prediction
    return prediction[0][0]


# function use to save figure image
def save_plot():
    # save the figure image
    fig.savefig('simple_regression.png')
    # returns to where it is called from
    return


# function to call prediction function when the square foot slider is changed
def predict_x1(value):
    # make the variables accessible from outside the function
    global line_1, line_2, plotCount
    # check if the slider value is greater or equal to the least x-input variable in the dataset
    if value >= np.min(np.array(X)):
        # Plot remove routine that removes the previous plotted lines in every 2 counts
        plotCount = plotCount + 1
        if plotCount == 2:
            plotCount = 1
            l1 = line_1.pop(0)
            l2 = line_2.pop(0)
            # remove the vertical and horizontal line tracker
            l1.remove()
            l2.remove()
        # call the model prediction function using the new value of the slider position
        prediction = predict_price(value)
        # move the new plot up and sideways so that it's not on top of the slider
        plt.subplots_adjust(left=0.2, bottom=0.4)
        # Plot Horizontal line
        line_1 = axa.plot([0, np.max(np.array(X))], [prediction, prediction], color='grey', linestyle='dashed')
        # Plot Vertical Line
        line_2 = axa.plot([value, value], [0, np.max(np.array(y)) + 1000000], color='grey', linestyle='dashed')
        # save plot image
        save_plot()
        return

    # perform this if slider value is less than the least x-input variable in the dataset
    else:
        # display a warning text when user slide to a value less than the least x-input variable in the dataset
        display_text = 'The least-square foot value should be >= ' + str(np.min(np.array(X))) + '\n'\
                       + '(the least value in the dataset used for training)'
        # display warning text
        axa.set_title(display_text, fontsize=12, fontweight='bold')
    # returns where it was called from
    return


#
# CONFIGURE THE SLIDER AXES AND SLIDER
#
# Set the slider position on the plot
x1SliderDim = plt.axes([axa.get_position().x0, axa.get_position().y0-0.12, axa.get_position().width, 0.03],
                       facecolor='lightgoldenrodyellow')
# display model performance at the bottom of the figure
text = fig.text(0.50, 0.02, result_text, fontsize=12, horizontalalignment='center', wrap=True, bbox=props)
# Make a horizontal slider to select square footage of home value
x1Slider = Slider(
    ax=x1SliderDim,
    label='Set square foot value \n (move slider to set):',  # set the slider title
    valinit=round(np.min(np.array(X))),  # set the initial slider position using the least value of X
    valmin=0,  # set the minimum slider value
    valstep=10,  # set the step value for each slider movement
    valmax=np.max(np.array(X))  # set the maximum slider value using the maximum value of X variable
)
# Register the update function with each slider
x1Slider.on_changed(predict_x1)
# set the slider to its initial position using the least value of X
get_least_x_value = round(np.min(np.array(X)))
# call the prediction function
predict_x1(get_least_x_value)
#
# PLOT THE HEATMAP ON A SUBPLOT
#
# create a subplot with a single figure
axb = fig.add_subplot(122)
# select some columns to visualize the data using heatmap
dff = df[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'grade']]  # or you can use dff = df[:, [0,1,2,3,9]]
# title by setting initial sizes
axb.set_title('Heatmap Data Visualization', fontsize=14, fontweight='bold')
# plot the heatmap
sns.heatmap(dff.corr(), annot=True)

# show the plot
plt.show()

