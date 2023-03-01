pipeline {
  agent {
    node { label 'joligan' }
    dockerfile {
      filename 'docker/Dockerfile.devel'
      additionalBuildArgs '--no-cache'
      args '-u jenkins'
    }

  }
  stages {
    stage('Tests') {
      steps {
        sh 'printenv | sort'
        sh 'mkdir /home/jenkins/app/checkpoints'
        sh '''
TORCH_HOME=/home/jenkins/app/.cache/ bash ./scripts/run_tests.sh /home/jenkins/app/checkpoints/'''
      }
    }

  }
  environment {
    DOCKER_PARAMS = '"--runtime nvidia -u jenkins"'
  }
}
