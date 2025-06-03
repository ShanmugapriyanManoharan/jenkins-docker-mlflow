pipeline {
    agent any

    stages {
        stage('Getting Project from Git') {
            steps {
                echo 'Project is downloading...'
                git branch: 'master', url: 'https://github.com/03sarath/jenkins-mlflow-docker.git'
            }
        }
        stage('Building Docker Container') {
            steps {
                sh 'docker build -t heartdisease-model .'
                sh 'docker run -d --name model heartdisease-model'
            }
        }
        stage('Preprocessing Stage') {
            steps {
                sh 'docker container exec model python3 preprocessing.py'
            }
        }
        stage('Training Stage') {
            steps {
                sh 'docker container exec model python3 train.py'
            }
        }
        stage('Test Stage') {
            steps {
                sh 'docker container exec model python3 train.py'
                sh 'docker container exec model python3 test.py'
                sh 'docker rm -f model'
            }
        }
    }
}
