# FastAPI Counter Application

This is a simple FastAPI-based web application that allows you to increase and view the current value of a counter. The counter value is stored in memory and can be manipulated using the provided API endpoints and a basic HTML frontend.

## Features

- Increase the counter value.
- Get the current counter value.
- Simple frontend with buttons for user interaction

## Technologies Used

- **Backend:** FastAPI
- **Frontend:** HTML and JavaScript
- **Containerization:** Docker

## Getting Started

Follow the steps below to run the project locally.

### Prerequisites

- **Python 3.9+**
- **Docker** (for containerization)

### Installation

1. docker build -t counter-app .
2. docker run -p 8000:8000 counter-app
