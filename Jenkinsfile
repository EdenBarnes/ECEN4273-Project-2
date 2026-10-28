pipeline {
    agent { label 'python' }

    options {
        timestamps()
        timeout(time: 15, unit: 'MINUTES')
        disableConcurrentBuilds(abortPrevious: true)
        buildDiscarder(logRotator(numToKeepStr: '20'))
    }

    environment {
        PIP_DISABLE_PIP_VERSION_CHECK = '1'
        VENV = "${WORKSPACE}/.venv"
    }

    stages {
        stage('Setup') {
            steps {
                sh '''
                    python -m venv "$VENV"
                    . "$VENV/bin/activate"
                    pip install -r requirements-dev.txt
                    pip install -e .
                '''
            }
        }
        stage('Build') {
            steps {
                echo 'Building..'
            }
        }
        stage('Test') {
            steps {
                echo 'Testing..'
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying..'
            }
        }
    }
}