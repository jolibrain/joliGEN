pipeline {
  agent {
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
  post {
      always {
      cleanWs(cleanWhenAborted: true, cleanWhenFailure: true, cleanWhenNotBuilt: true, cleanWhenSuccess: true, cleanWhenUnstable: true, cleanupMatrixParent: true, deleteDirs: true)
    }
    success {
      catchError {
        rocketSend(channel: 'build', message: 'Build succeed' ,color: 'green' )
      }
    }
    aborted {
      catchError {
        rocketSend(channel: 'build', message: 'Build superseded or aborted')
      }
    }
    unstable {
      catchError {
        rocketSend(channel: 'build', message: 'Build failed', color: 'red')
      }
    }
    failure {
      catchError {
        rocketSend(channel: 'build', message: 'Build failed', color: 'red')
      }
    }
  }
  environment {
    DOCKER_PARAMS = '"--runtime nvidia -u jenkins"'
  }
}
