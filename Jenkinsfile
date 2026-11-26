pipeline {
    agent any

    environment {
        VENV_DIR = '.venv'
        DEPLOY_DIR = 'C:\\deploy\\SLAYORSKIPMOVIESENTIMENT'
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
                    if exist requirements.txt (
                        "%VENV_DIR%\\Scripts\\python.exe" -m pip install -r requirements.txt
                    )
                    "%VENV_DIR%\\Scripts\\python.exe" -m pip install pytest
                '''
            }
        }

        stage('Run Pytest') {
            steps {
                bat '''
                    "%VENV_DIR%\\Scripts\\python.exe" -m pytest -v --junitxml=test-results.xml
                '''
            }
        }

        stage('Deploy') {
            steps {
                bat '''
                    if not exist "%DEPLOY_DIR%" mkdir "%DEPLOY_DIR%"
                    robocopy . "%DEPLOY_DIR%" /MIR /XD .git .venv __pycache__ .pytest_cache
                    if %ERRORLEVEL% LEQ 7 exit /b 0
                '''
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true, testResults: 'test-results.xml'
        }
    }
}