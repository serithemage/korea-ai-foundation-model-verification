#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { LlmDnaStack } from '../lib/llm-dna-stack';

const app = new cdk.App();

// Multi-region capable. Pass CDK_TARGET_REGION=us-west-2 to deploy a separate stack there.
// Stack names: us-east-1 keeps the original short name; other regions get a region suffix
// to avoid CFN conflicts within the same account.
const region = process.env.CDK_TARGET_REGION ?? 'us-east-1';
const stackName = region === 'us-east-1' ? 'LlmDnaStack' : `LlmDnaStack-${region}`;

new LlmDnaStack(app, stackName, {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region,
  },
  description: `LLM-DNA lineage analysis (${region}): S3 cache, HF token secret, SageMaker exec role, ECR repo`,
});
