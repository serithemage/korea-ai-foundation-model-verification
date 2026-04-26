#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { LlmDnaStack } from '../lib/llm-dna-stack';

const app = new cdk.App();

// Region is intentionally hard-coded: ml.p5.48xlarge spot quota and capacity (score 9)
// were verified specifically in us-east-1. Do not honor AWS_DEFAULT_REGION / profile region.
new LlmDnaStack(app, 'LlmDnaStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: 'us-east-1',
  },
  description: 'LLM-DNA lineage analysis: S3 cache, HF token secret, SageMaker exec role, ECR repo for spot training jobs',
});
