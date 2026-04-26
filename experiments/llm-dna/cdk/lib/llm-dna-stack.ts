import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';

export class LlmDnaStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const artifactsBucket = new s3.Bucket(this, 'Artifacts', {
      bucketName: `llm-dna-${this.account}-${this.region}`,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      versioned: false,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      lifecycleRules: [
        {
          id: 'expire-hf-cache',
          prefix: 'cache/',
          expiration: cdk.Duration.days(30),
        },
        {
          id: 'expire-incomplete-uploads',
          abortIncompleteMultipartUploadAfter: cdk.Duration.days(1),
        },
      ],
    });

    const hfTokenSecret = new secretsmanager.Secret(this, 'HfToken', {
      secretName: 'llm-dna/hf-token',
      description: 'Hugging Face access token used by SageMaker training jobs to download gated models',
    });

    const containerRepo = new ecr.Repository(this, 'Container', {
      repositoryName: 'llm-dna',
      imageScanOnPush: true,
      emptyOnDelete: false,
      lifecycleRules: [
        {
          description: 'Keep only the 5 most recent images',
          maxImageCount: 5,
        },
      ],
    });

    const sagemakerRole = new iam.Role(this, 'SageMakerExecRole', {
      roleName: 'LlmDnaSageMakerExecutionRole',
      assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
      description: 'SageMaker training job execution role for LLM-DNA extraction',
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess'),
      ],
    });
    artifactsBucket.grantReadWrite(sagemakerRole);
    hfTokenSecret.grantRead(sagemakerRole);
    containerRepo.grantPull(sagemakerRole);

    cdk.Tags.of(this).add('project', 'korea-ai-foundation-model-verification');
    cdk.Tags.of(this).add('component', 'llm-dna');

    new cdk.CfnOutput(this, 'ArtifactsBucketName', {
      value: artifactsBucket.bucketName,
      description: 'S3 bucket for HF model cache and DNA vector outputs',
    });
    new cdk.CfnOutput(this, 'HfTokenSecretArn', {
      value: hfTokenSecret.secretArn,
      description: 'Set the HF token value with: aws secretsmanager put-secret-value --secret-id <arn> --secret-string <token>',
    });
    new cdk.CfnOutput(this, 'ContainerRepoUri', {
      value: containerRepo.repositoryUri,
      description: 'Push the llm-dna container image to this ECR repo',
    });
    new cdk.CfnOutput(this, 'SageMakerRoleArn', {
      value: sagemakerRole.roleArn,
      description: 'IAM role used by SageMaker training jobs',
    });
  }
}
