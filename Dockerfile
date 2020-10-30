FROM python:3.8-slim-buster

COPY ./requirements.txt ./
COPY ./setup.py ./

# Copy the latest version of the model
COPY ./bin/models/synsypa_transformer_2020-10-29_epoch200_loss0.18 /bin/models/
COPY ./src /src

# Install dependenceis
RUN apt-get -y update && \
    pip install -r requirements.txt && \
    apt-get -y autoremove

# Load package
RUN pip install -e .

# Change to working directory
WORKDIR /

# Run the shell file
CMD ["python", "src/discord_bot.py"]