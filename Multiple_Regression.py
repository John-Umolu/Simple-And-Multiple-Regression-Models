# import the python libraries
import pandas as pd  # use perform dataset analysis
import matplotlib.pyplot as plt  # use for creating plots
from matplotlib.widgets import Slider  # use to create a slider to help user choose any input variable
import numpy as np  # use to perform mathematical functions
from sklearn.linear_model import LinearRegression  # use to create and train the model
from sklearn.model_selection import train_test_split  # use to split data use for training and testing
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # use to evaluate models performance

# declared variables to use
plotCount: int = 0  # use for track slider movement to clear previous plot
line_1 = (0, 0)  # stores the last slider vertical line plot
line_2 = (0, 0)  # stores the last slider vertical line plot
value1 = 0  # stores the set X1 value use for prediction
value2 = 0  # stores the set X2 value use for prediction
value3 = 0  # stores the set X3 value use for prediction
value4 = 0  # stores the set X4 value use for prediction

# read dataset csv file
df = pd.read_csv('houseprice_data.csv')
# remove any null values from data rows
df = df.dropna()
# get the independent (input) variables
X = df[['sqft_living', 'grade', 'bedrooms', 'bathrooms']].values  # or you can use: X = df[:, [3, 9]].values
# get the dependent (target) variable
y = df[['price']].values  # or you can use: y = df[:, 0].values
# split data into test and train using 30% of data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=0)
# build the linear regression model
model = LinearRegression()
# train the linear regression model
model.fit(X_train, y_train)
# find the coefficients
coef_value = model.coef_[0]
print('Coefficients: ', ', '.join(str(round(e, 1)) for e in coef_value))
# find the intercept
intercept = model.intercept_
print('Intercept: ', round(intercept[0], 1))
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
# GET ALL THE BUILT MODEL EVALUATIONS RESULTS
#
# string all the model performance text with each on a newline
result_text = 'Evaluating the multiple regression model: \n' \
              + 'Coefficients: ' + ', '.join(str(round(e, 1)) for e in coef_value) + '\n' \
              + 'Intercept: ' + ', '.join(str(round(e, 1)) for e in intercept) + '\n' \
              + str('Mean squared error (Training): %.1f' % mean_squared_error(y_train, model.predict(X_train))) + '\n' \
              + str('Mean squared error (Testing): %.1f' % mean_squared_error(y_test, model.predict(X_test))) + '\n' \
              + str('Mean absolute error: %.1f' % mean_absolute_error(y_test, model.predict(X_test))) + '\n' \
              + str('Coefficient of determination (R^2 score): '
                    + str(round(r2_score(y_test, model.predict(X_test)) * 100, 2)) + ' %' + '\n'
                    + 'Model training Score: ' + str(round(model.score(X_train, y_train) * 100, 1)) + ' %' + '\n'
                    + 'Model testing Score: ' + str(round(model.score(X_test, y_test) * 100, 1)) + ' %')
# create the new figure with width = 15 and height = 7
fig = plt.figure(figsize=(15, 7))
# title by setting initial sizes
fig.suptitle('Multiple Regression By Umolu John Chukwuemeka (2065655)', color='blue', fontsize=16, fontweight='bold')
# create box to contain the model performance text using the matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# set the figure title
fig.canvas.manager.set_window_title('Coursework Task 1: Simple Regression By Umolu John Chukwuemeka (2065655)')
#
# FIRST 3D PLOT
#
# create a subplot with a single figure
axa = fig.add_subplot(121, projection='3d')
# plot the scatter plot of x and y variables
axa.scatter(X[:, 0], X[:, 1], y, color='blue')
# draw the regression line
axa.plot(X[:, 0], model.predict(X), color='red', label=' Predicted Regression line')
# set the x-axis label
axa.set_xlabel('Square foot')
# set the y-axis label
axa.set_ylabel('Grade')
# set the z-axis label
axa.set_zlabel('Price')
# set the minimum and maximum range for x-axis
axa.set_xlim(0, np.max(np.array(X[:, 0])))
# set the minimum and maximum range for y-axis
axa.set_ylim(0, np.max(np.array(X[:, 1])))
# set the minimum and maximum range for z-axis
axa.set_zlim(0, np.max(np.array(y)) + 1000000)
#
# SECOND 3D PLOT
#
# create a subplot with a single figure
axb = fig.add_subplot(122, projection='3d')
# plot the scatter plot of x and y variables
axb.scatter(X[:, 2], X[:, 3], y, color='blue')
# draw the regression line
axb.plot(X[:, 2], model.predict(X), color='red', label=' Predicted Regression line')
# set the x-axis label
axb.set_xlabel('Bedrooms')
# set the y-axis label
axb.set_ylabel('Bathrooms')
# set the z-axis label
axb.set_zlabel('Price')
# set the minimum and maximum range for x-axis
axb.set_xlim(0, np.max(np.array(X[:, 2])))
# set the minimum and maximum range for y-axis
axb.set_ylim(0, np.max(np.array(X[:, 3])))
# set the minimum and maximum range for z-axis
axb.set_zlim(0, np.max(np.array(y)) + 1000000)

# move the figure up so that it's not on top of the slider
fig.subplots_adjust(bottom=0.45)


# function use to save figure image
def save_plot():
    # save the figure image
    fig.savefig('multiple_regression.png')
    # returns to where it is called from
    return


# function used to predict values
def predict_price(value1, value2, value3, value4):
    # make the variables accessible from outside the function
    global input_variables, prediction, display_text
    # get slider value and convert it to numpy array
    input_variables = np.array([[value1, value2, value3, value4]])
    # get the predicted price value using the built model
    prediction = model.predict(input_variables)
    # variable to store all the square foot and prediction text to be displayed
    display_text = 'Predicted Price of the house: ' + str(round(prediction[0][0], 1))
    # display the square foot and prediction text using figure subtitle
    axa.set_title(display_text, fontsize=12, fontweight='bold')
    # display the square foot and prediction text using figure subtitle on console
    print(display_text)
    # save plot image
    save_plot()
    # returns to where it was called from
    return


# function to call prediction function when the square foot slider is changed
def predict_x1(slider1_value):
    # make the variables accessible from outside the function
    global line_1, line_2, plotCount, value1, value2, value3, value4
    value1 = slider1_value
    # check if grade value is less than or equal to zero
    if value2 <= 0:
        # set grade value to the least value in the dataset
        value2 = round(np.max(np.array(X[:, 1]))/2)
    if value3 <= 0:
        # set grade value to the least value in the dataset
        value3 = round(np.max(np.array(X[:, 2]))/2)
    if value4 <= 0:
        # set grade value to the least value in the dataset
        value4 = round(np.max(np.array(X[:, 3]))/2)
    # call the prediction function
    predict_price(value1, value2, value3, value4)
    # returns to where it was called from
    return


# function to call prediction function when the square foot slider is changed
def predict_x2(slider2_value):
    # make the variables accessible from outside the function
    global line_1, line_2, plotCount, value1, value2, value3, value4
    value2 = slider2_value
    # check if grade value is less than or equal to zero
    if value1 <= 0:
        # set grade value to the least value in the dataset
        value1 = round(np.max(np.array(X[:, 0]))/2)
    if value3 <= 0:
        # set grade value to the least value in the dataset
        value3 = round(np.max(np.array(X[:, 2]))/2)
    if value4 <= 0:
        # set grade value to the least value in the dataset
        value4 = round(np.max(np.array(X[:, 3]))/2)
    # call the prediction function
    predict_price(value1, value2, value3, value4)
    # returns to where it was called from
    return


# function to call prediction function when the square foot slider is changed
def predict_x3(slider3_value):
    # make the variables accessible from outside the function
    global line_1, line_2, plotCount, value1, value2, value3, value4
    value3 = slider3_value
    # check if grade value is less than or equal to zero
    if value1 <= 0:
        # set grade value to the least value in the dataset
        value1 = round(np.max(np.array(X[:, 0]))/2)
    if value2 <= 0:
        # set grade value to the least value in the dataset
        value2 = round(np.max(np.array(X[:, 1]))/2)
    if value4 <= 0:
        # set grade value to the least value in the dataset
        value4 = round(np.max(np.array(X[:, 3]))/2)
    # call the prediction function
    predict_price(value1, value2, value3, value4)
    # returns to where it was called from
    return


# function to call prediction function when the square foot slider is changed
def predict_x4(slider4_value):
    # make the variables accessible from outside the function
    global line_1, line_2, plotCount, value1, value2, value3, value4
    value4 = slider4_value
    # check if grade value is less than or equal to zero
    if value1 <= 0:
        # set grade value to the least value in the dataset
        value1 = round(np.max(np.array(X[:, 0]))/2)
    if value2 <= 0:
        # set grade value to the least value in the dataset
        value2 = round(np.max(np.array(X[:, 1]))/2)
    if value3 <= 0:
        # set grade value to the least value in the dataset
        value3 = round(np.max(np.array(X[:, 2]))/2)
    # call the prediction function
    predict_price(value1, value2, value3, value4)
    # returns to where it was called from
    return


#
# CONFIGURE THE SLIDERS
#
# Set the sliders axes position on the figure
x1SliderDim = plt.axes([axa.get_position().x0, axa.get_position().y0-0.1, axa.get_position().width, 0.03],
                       facecolor='lightgoldenrodyellow')
x2SliderDim = plt.axes([axa.get_position().x0, axa.get_position().y0-0.15, axa.get_position().width, 0.03],
                       facecolor='lightgoldenrodyellow')
x3SliderDim = plt.axes([axb.get_position().x0, axb.get_position().y0-0.1, axb.get_position().width, 0.03],
                       facecolor='lightgoldenrodyellow')
x4SliderDim = plt.axes([axb.get_position().x0, axb.get_position().y0-0.15, axb.get_position().width, 0.03],
                       facecolor='lightgoldenrodyellow')
# display model performance at the bottom of the figure
text = fig.text(0.50, 0.02, result_text, fontsize=12, horizontalalignment='center', wrap=True, bbox=props)
# Make a horizontal slider to select square footage of home value
x1Slider = Slider(
    ax=x1SliderDim,
    label='Set Sq-ft:',  # set the slider title
    valinit=round(np.max(np.array(X[:, 0]))/2),  # set the initial slider position using the least value of X
    valmin=0,  # set the minimum slider value
    valstep=10,  # set the step value for each slider movement
    valmax=np.max(np.array(X[:, 0]))  # set the maximum slider value using the maximum value of X variable
)
# Make a horizontal slider to select square footage of home value
x2Slider = Slider(
    ax=x2SliderDim,
    label='Set Grade:',  # set the slider title
    valinit=round(np.max(np.array(X[:, 1]))/2),  # set the initial slider position using the least value of X
    valmin=0,  # set the minimum slider value
    valstep=1,  # set the step value for each slider movement
    valmax=np.max(np.array(X[:, 1]))  # set the maximum slider value using the maximum value of X variable
)
x3Slider = Slider(
    ax=x3SliderDim,
    label='Set Bedrooms:',  # set the slider title
    valinit=round(np.max(np.array(X[:, 2]))/2),  # set the initial slider position using the least value of X
    valmin=0,  # set the minimum slider value
    valstep=1,  # set the step value for each slider movement
    valmax=np.max(np.array(X[:, 2]))  # set the maximum slider value using the maximum value of X variable
)
# Make a horizontal slider to select square footage of home value
x4Slider = Slider(
    ax=x4SliderDim,
    label='Set Bathrooms:',  # set the slider title
    valinit=round(np.max(np.array(X[:, 3]))/2),  # set the initial slider position using the least value of X
    valmin=0,  # set the minimum slider value
    valstep=1,  # set the step value for each slider movement
    valmax=np.max(np.array(X[:, 3]))  # set the maximum slider value using the maximum value of X variable
)
# Register the update function with each slider
x1Slider.on_changed(predict_x1)
# Register the update function with each slider
x2Slider.on_changed(predict_x2)
# Register the update function with each slider
x3Slider.on_changed(predict_x3)
# Register the update function with each slider
x4Slider.on_changed(predict_x4)

# get the max value and call the prediction function
predict_x1(round(np.max(np.array(X[:, 0]))/2))

# show the plot
plt.show()


