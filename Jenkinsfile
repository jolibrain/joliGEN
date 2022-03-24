pipeline {
  agent {
    dockerfile {
      filename 'docker/Dockerfile.build'
      additionalBuildArgs '--no-cache'
      args '-u root'
    }

  }
  stages {
    stage('Tests') {
      steps {
        sh 'mkdir checkpoints'
        sh '''
TORCH_HOME=/app/.cache/ bash ./scripts/run_tests.sh checkpoints/'''
      }
    }

  }
  environment {
    DOCKER_PARAMS = '"--runtime nvidia -u root"'
  }
}