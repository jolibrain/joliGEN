pipeline {
  agent {
    dockerfile {
      filename 'docker/Dockerfile.build'
      additionalBuildArgs '--no-cache'
    }

  }
  stages {
    stage('Tests') {
      steps {
        sh 'mkdir checkpoints'
        sh '''
bash ./scripts/run_tests.sh checkpoints/'''
      }
    }

  }
  environment {
    DOCKER_PARAMS = '"--runtime nvidia"'
  }
}