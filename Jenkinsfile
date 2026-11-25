pipeline {
    agent any

    environment {
        VENV_DIR = '.venv'
    }

    stages {
        stage('Setup Python Venv') {
            steps {
                bat '''
                    if not exist "%VENV_DIR%\\Scripts\\python.exe" (
                        python -m venv "%VENV_DIR%"
                    )
                '''
            }
        }

        stage('Install Requirements') {
            steps {
                bat '''
                    "%VENV_DIR%\\Scripts\\python.exe" -m pip install --upgrade pip
                    "%VENV_DIR%\\Scripts\\pip.exe" install -r requirements.txt pytest
                '''
            }
        }

        stage('Run Pytest') {
            steps {
                bat '''
                    "%VENV_DIR%\\Scripts\\python.exe" -m pytest -v
                '''
            }
        }
    }
}
