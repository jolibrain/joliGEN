pipeline {
  agent {
    dockerfile {
      filename 'docker/Dockerfile.devel'
      additionalBuildArgs '--no-cache'
      args '--shm-size=8gb -u jenkins'
    }

  }
  stages {
    stage('Tests') {
      when {
       expression {!env.CHANGE_ID || pullRequest.labels.findAll { it == "ci:skip-tests" }.size() == 0 }
      }
      steps {
        lock(resource: null, label: "${NODE_NAME}-gpu", variable: 'LOCKED_GPU', quantity: 1) {
          sh 'printenv | sort'
          sh 'mkdir /home/jenkins/app/checkpoints'
          sh '''
          export CUDA_VISIBLE_DEVICES=$(echo ${LOCKED_GPU} | sed -n -e "s/[^,]* GPU \\([^[0-9,]]\\)*/\\1/gp")
          echo "Running on GPU ${CUDA_VISIBLE_DEVICES}"
          TORCH_HOME=/home/jenkins/app/.cache/ TORCH_CUDA_ARCH_LIST="8.6" bash ./scripts/run_tests.sh /home/jenkins/app/checkpoints/'''
        }
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
    DOCKER_PARAMS = '"--runtime nvidia --shm-size=8gb -u jenkins"'
  }
}
