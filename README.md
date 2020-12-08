# Movie Recommender System

Project 9 for the Data Science Bootcamp at SPICED Academy Berlin, by Lorena Ciutacu and Daniela Morariu.

<img src="https://github.com/lorenanda/movie-recommender/blob/main/demo.gif" width="700" height="350">

## Description
The days of endless indecissive scrolling are over! We created a recommender system with a web interface that recommends you new movies based on your preferences. You only need to rate up to 15 movies and specify your preference for old vs. new movies, then you get a list of 5 movies along with their plot summary and trailer. We built this with:
- Python
- pandas, scikit-learn (NMF), surprise (SVD)
- HTML, CSS, Flask
- TMBD API

## How to use
In a terminal:
1. Clone this repo: `git clone https://github.com/lorenanda/movie-recommender.git`
2. Install the necessary libraries: `pip install -r requirements.txt`
3. Change the directory to the main application `cd flask_app`
4. Get your API key from [TMDB](https://developers.themoviedb.org/3/getting-started/introduction) and write it in `config.py`
5. Run these three commands:
    - `export FLASK_APP=application.py`
    - `export FLASK_DEBUG=1`
    - `flask run`
6. Open the listed localhost http://127.0.0.1:5000/ in a browser.
7. Grab some popcorn, lean back, and enjoy a recommended movie!

*The project was tested on Chrome and Firefox!*
